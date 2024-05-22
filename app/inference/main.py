"""Inference Pipeline Module.

This module implements the inference pipeline for bird sound classification.
It listens for messages from a RabbitMQ queue, fetches the corresponding audio files from MinIO,
performs inference using a pre-trained model, and publishes the results back to another RabbitMQ queue.

The module relies on the following dependencies:
- `app_utils.minio`: Provides utility functions for interacting with MinIO.
- `app_utils.rabbitmq`: Provides utility functions for interacting with RabbitMQ.
- `minio`: A library for interacting with MinIO object storage.
- `model_serve.model_serve`: Provides the `ModelServer` class for loading and running the pre-trained model.
- `src.models.bird_dict`: Provides a dictionary mapping bird species to their corresponding labels.

Example usage:
1. Set the required environment variables for RabbitMQ, MinIO, and model paths.
2. Run the script: `python inference_pipeline.py`
   The script will start listening for messages from the specified RabbitMQ queue and process them accordingly.

Note: Make sure to have the necessary dependencies installed and the pre-trained model available before running the script.

"""

import json
import logging
import os

from app_utils.minio import write_file_to_minio
from app_utils.rabbitmq import (
    consume_messages,
    get_rabbit_connection,
    publish_message,
)
from minio import Minio
from model_serve.model_serve import ModelServer

from src.models.bird_dict import BIRD_DICT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


#################### CONFIG ####################
WEIGHTS_PATH = "models/detr_noneg_100q_bs20_r50dc5"
TEST_FILE_PATH = "inference/Turdus_merlula.wav"

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
FORWARDING_QUEUE = os.getenv("RABBITMQ_QUEUE_API2INF")
FEEDBACK_QUEUE = os.getenv("RABBITMQ_QUEUE_INF2API")

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
MINIO_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
MINIO_BUCKET = os.getenv("MINIO_BUCKET")

#################### STORAGE ####################
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,
)


#################### QUEUE ####################
def callback(body) -> None:
    """Callback function to handle messages from RabbitMQ.

    Args:
        body (bytes): The message body received from RabbitMQ.

    Returns:
        None

    """
    message = json.loads(body.decode())
    minio_path = message["minio_path"]
    email = message["email"]
    ticket_number = message["ticket_number"]

    logger.info(
        "Received message from RabbitMQ: MinIO path=%s, Email=%s, Ticket number=%s", minio_path, email, ticket_number
    )
    run_inference_pipeline(minio_path, email, ticket_number)


#################### ML I/O  ####################
def run_inference_pipeline(minio_path, email, ticket_number) -> None:
    """Run the inference pipeline for a given audio file.

    Args:
        minio_path (str): The path of the audio file in MinIO.
        email (str): The email address associated with the inference request.
        ticket_number (str): The ticket number associated with the inference request.

    Returns:
        None

    """
    file_name = os.path.basename(minio_path)
    local_file_path = "/tmp/%s" % file_name  # Temporary local file path

    # Fetch the WAV file from MinIO and save it locally
    try:
        minio_client.fget_object(MINIO_BUCKET, file_name, local_file_path)
        logger.info("WAV file downloaded from MinIO: %s", file_name)
    except Exception as e:
        logger.error("Error downloading WAV file from MinIO: %s", str(e))
        return

    inference = ModelServer(WEIGHTS_PATH, BIRD_DICT)
    inference.load()
    output = inference.get_classification(local_file_path)
    logger.info("Classification output: %s", output)

    json_file_name = os.path.splitext(file_name)[0] + ".json"
    json_output = list(output.values())[0]  # Extract the JSON output from the dictionary

    # Write the JSON output to MinIO using the helper function
    json_data = json.dumps(json_output).encode("utf-8")
    write_file_to_minio(minio_client, MINIO_BUCKET, json_file_name, json_data)

    # Publish the message containing the MinIO paths, email, and ticket number on the feedback channel
    message = {
        "wav_minio_path": "%s/%s" % (MINIO_BUCKET, file_name),
        "json_minio_path": json_file_name,
        "email": email,
        "ticket_number": ticket_number,
    }
    publish_message(rabbitmq_channel, FEEDBACK_QUEUE, message)


#################### MAIN LOOP ####################
if __name__ == "__main__":
    rabbitmq_connection = get_rabbit_connection(RABBITMQ_HOST, RABBITMQ_PORT)
    rabbitmq_channel = rabbitmq_connection.channel()
    rabbitmq_channel.queue_declare(queue=FORWARDING_QUEUE)

    logging.info("Declaring queue: %s", FEEDBACK_QUEUE)
    rabbitmq_channel.queue_declare(queue=FEEDBACK_QUEUE)

    logger.info("Waiting for messages from queue: %s", FORWARDING_QUEUE)
    consume_messages(rabbitmq_channel, FORWARDING_QUEUE, callback)
