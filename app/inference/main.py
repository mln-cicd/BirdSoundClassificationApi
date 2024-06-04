import json
import logging
import os

from app_utils.minio import write_file_to_minio
from app_utils.rabbitmq import consume_messages, get_rabbit_connection, publish_message
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
    """Trigger an inference pipeline run as RabbitMQ message callback."""
    message = json.loads(body.decode())
    minio_path = message["minio_path"]
    email = message["email"]
    ticket_number = message["ticket_number"]

    logger.info(
        f"Received message from RabbitMQ: MinIO path={minio_path}, "
        f"Email={email}, Ticket number={ticket_number}"
    )
    run_inference_pipeline(minio_path, email, ticket_number)


def download_wav_file(minio_client, bucket, file_name, local_file_path):
    try:
        minio_client.fget_object(bucket, file_name, local_file_path)
        logger.info(f"WAV file downloaded from MinIO: {file_name}")
    except Exception as e:
        logger.error(f"Error downloading WAV file from MinIO: {e!s}")
        raise


def perform_inference(weights_path, bird_dict, local_file_path):
    inference = ModelServer(weights_path, bird_dict)
    inference.load()
    output = inference.get_classification(local_file_path)
    logger.info(f"Classification output: {output}")
    return output


def save_classification_to_minio(minio_client, bucket, file_name, output):
    json_file_name = os.path.splitext(file_name)[0] + ".json"
    json_output = next(iter(output.values()))  # Extract the JSON output from the dict
    json_data = json.dumps(json_output).encode("utf-8")
    write_file_to_minio(minio_client, bucket, json_file_name, json_data)
    return json_file_name


def publish_feedback_message(
    rabbitmq_channel, feedback_queue, file_name, json_file_name, email, ticket_number
):
    message = {
        "wav_minio_path": f"{MINIO_BUCKET}/{file_name}",
        "json_minio_path": f"{json_file_name}",
        "email": email,
        "ticket_number": ticket_number,
    }
    publish_message(rabbitmq_channel, feedback_queue, message)


def run_inference_pipeline(minio_path, email, ticket_number) -> None:
    """Run inference pipeline, output classification and publish feedback message."""
    file_name = os.path.basename(minio_path)
    local_file_path = f"/tmp/{file_name}"  # Temporary local file path

    # Fetch the WAV file from MinIO and save it locally
    download_wav_file(minio_client, MINIO_BUCKET, file_name, local_file_path)

    # Perform inference
    output = perform_inference(WEIGHTS_PATH, BIRD_DICT, local_file_path)

    # Save classification results to MinIO
    json_file_name = save_classification_to_minio(
        minio_client, MINIO_BUCKET, file_name, output
    )

    # Publish feedback message
    publish_feedback_message(
        rabbitmq_channel,
        FEEDBACK_QUEUE,
        file_name,
        json_file_name,
        email,
        ticket_number,
    )


#################### MAIN LOOP ####################
if __name__ == "__main__":
    rabbitmq_connection = get_rabbit_connection(RABBITMQ_HOST, RABBITMQ_PORT)
    rabbitmq_channel = rabbitmq_connection.channel()
    rabbitmq_channel.queue_declare(queue=FORWARDING_QUEUE)

    logging.info(f"Declaring queue: {FEEDBACK_QUEUE}")
    rabbitmq_channel.queue_declare(queue=FEEDBACK_QUEUE)

    logger.info(f"Waiting for messages from queue: {FORWARDING_QUEUE}")
    consume_messages(rabbitmq_channel, FORWARDING_QUEUE, callback)
