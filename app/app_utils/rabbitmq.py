"""RabbitMQ Utility Module.

This module provides utility functions for interacting with RabbitMQ message broker. It
includes functions for connecting to RabbitMQ, publishing messages, consuming messages,
and processing feedback messages.

"""

import asyncio
import json
import logging
import time

import pika
from app_utils.smtplib import send_email

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Global variable to manage RabbitMQ connection
RABBIT_CONNECTION = None


def connect_to_rabbitmq(host, port, max_retries=5, retry_delay=5) -> pika.BlockingConnection:
    """Connects to RabbitMQ with automatic retries.

    Args:
        host (str): The hostname or IP address of the RabbitMQ server.
        port (int): The port number of the RabbitMQ server.
        max_retries (int, optional): The maximum number of connection retries.
        retry_delay (int, optional): The delay in seconds between each retry attempt.

    Returns:
        pika.BlockingConnection: The established connection to RabbitMQ.

    Raises:
        Exception: If the connection to RabbitMQ fails after the specified number of retries.

    """
    global RABBIT_CONNECTION
    retry_count = 0

    while retry_count < max_retries:
        logging.info("Attempting to connect to RabbitMQ (Attempt %d/%d)", retry_count + 1, max_retries)
        try:
            RABBIT_CONNECTION = pika.BlockingConnection(pika.ConnectionParameters(host=host, port=port))
            logging.info("Successfully connected to RabbitMQ")
            return RABBIT_CONNECTION
        except pika.exceptions.AMQPConnectionError as error:
            logging.error("Error connecting to RabbitMQ: %s. Retrying in %d seconds...", str(error), retry_delay)
            retry_count += 1
            time.sleep(retry_delay)
            if retry_count == max_retries:
                logging.critical("Failed to connect to RabbitMQ after multiple retries.")
                raise Exception("Failed to connect to RabbitMQ after multiple retries.") from error


def get_rabbit_connection(host, port) -> pika.BlockingConnection:
    """Gets the current RabbitMQ connection or establishes a new one if necessary.

    Returns:
        pika.BlockingConnection: The active RabbitMQ connection.

    """
    global RABBIT_CONNECTION
    logging.info("Checking RabbitMQ connection")
    if RABBIT_CONNECTION is None or RABBIT_CONNECTION.is_closed:
        logging.info("RabbitMQ connection is None or closed. Establishing a new connection.")
        connect_to_rabbitmq(host, port)
        logging.info("RabbitMQ connection established.")
    else:
        logging.info("RabbitMQ connection is active.")
    return RABBIT_CONNECTION


def publish_minio_path(channel, queue_name, minio_path) -> None:
    """Publishes a MinIO path to a specified RabbitMQ queue.

    Args:
        channel: The active channel of the RabbitMQ connection.
        queue_name (str): The name of the queue where the message will be published.
        minio_path (str): The MinIO path to be published.

    Returns:
        None

    """
    logging.info("Preparing to publish MinIO path to queue: %s", queue_name)
    try:
        channel.basic_publish(exchange="", routing_key=queue_name, body=minio_path)
        logging.info("Published MinIO path: %s", minio_path)
    except pika.exceptions.AMQPError as error:
        logging.error("Failed to publish MinIO path: %s", str(error))
        raise Exception("Failed to publish MinIO path") from error


def publish_message(channel, queue_name, message) -> None:
    """Publishes a message to a specified RabbitMQ queue.

    Args:
        channel: The active channel of the RabbitMQ connection.
        queue_name (str): The name of the queue where the message will be published.
        message (dict): The message to be published, containing the MinIO path, email, and ticket number.

    Returns:
        None

    """
    logging.info("Preparing to publish message to queue: %s", queue_name)
    try:
        channel.basic_publish(exchange="", routing_key=queue_name, body=json.dumps(message))
        logging.info("Published message: %s", message)
    except pika.exceptions.AMQPError as error:
        logging.error("Failed to publish message: %s", str(error))


def consume_messages(channel, queue_name, callback) -> None:
    """Consume messages from a specified RabbitMQ queue and invokes a callback function
    for each message.

    Args:
        channel: The active channel of the RabbitMQ connection.
        queue_name (str): The name of the queue to consume messages from.
        callback (function): The callback function to be invoked for each received message.
                             The function should accept a single argument, which is the message body.

    Returns:
        None

    """

    def on_message(channel, method, _, body):
        """Callback function invoked for each received message.

        Args:
            channel: The channel object.
            method: The delivery method.
            body: The message body.

        """
        callback(body)
        channel.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_consume(queue=queue_name, on_message_callback=on_message)
    channel.start_consuming()


def process_feedback_message(body, minio_client, minio_bucket) -> None:
    """Process a feedback message received from RabbitMQ.

    This function extracts the email, JSON MinIO path, and ticket number from the message body.
    It then sends an email using the extracted information and the provided MinIO client and bucket.

    Args:
        body (bytes): The body of the feedback message received from RabbitMQ.
        minio_client (Minio): The MinIO client instance used to interact with MinIO.
        minio_bucket (str): The name of the MinIO bucket where the JSON file is stored.

    Returns:
        None

    """
    data = json.loads(body)
    email = data["email"]
    json_minio_path = data["json_minio_path"]
    ticket_number = data["ticket_number"]

    json_file_path = f"{json_minio_path}"
    send_email(email, json_file_path, ticket_number, minio_client, minio_bucket)


async def consume_feedback_messages(rabbitmq_channel, feedback_queue, minio_client, minio_bucket) -> None:
    """Consume feedback messages from the specified RabbitMQ queue.

    This function continuously consumes feedback messages from the specified RabbitMQ queue.
    For each received message, it processes the message using the `process_feedback_message` function
    and acknowledges the message. If no message is available, it waits for 1 second before checking again.

    Args:
        rabbitmq_channel (pika.adapters.blocking_connection.BlockingChannel): The RabbitMQ channel used for consuming messages.
        feedback_queue (str): The name of the RabbitMQ queue to consume feedback messages from.
        minio_client (Minio): The MinIO client instance used to interact with MinIO.
        minio_bucket (str): The name of the MinIO bucket where the JSON files are stored.

    Returns:
        None

    """
    while True:
        method_frame, _, body = rabbitmq_channel.basic_get(queue=feedback_queue)
        if method_frame:
            process_feedback_message(body, minio_client, minio_bucket)
            rabbitmq_channel.basic_ack(delivery_tag=method_frame.delivery_tag)
        else:
            await asyncio.sleep(1)  # Wait for 1 second before checking again
