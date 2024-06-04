import json
import time
from unittest.mock import MagicMock, patch

import pytest

from app.inference.main import (
    download_wav_file,
    perform_inference,
    publish_feedback_message,
    run_inference_pipeline,
    save_classification_to_minio,
)
from tests.utils.docker import start_inference_container


@pytest.fixture(scope="session")
def inference_container():
    container = start_inference_container()
    yield container
    container.stop()
    container.remove()


@pytest.mark.usefixtures("inference_container")
def test_download_wav_file(mock_minio_client):
    mock_minio_client.fget_object.return_value = None
    download_wav_file(mock_minio_client, "bucket", "file.wav", "/tmp/file.wav")
    mock_minio_client.fget_object.assert_called_once_with(
        "bucket", "file.wav", "/tmp/file.wav"
    )


@pytest.mark.usefixtures("inference_container")
@patch("app.inference.main.ModelServer")
def test_perform_inference(mock_model_server):
    mock_model = mock_model_server.return_value
    mock_model.get_classification.return_value = {"bird": "species"}
    output = perform_inference("weights_path", {"bird": "species"}, "/tmp/file.wav")
    mock_model.load.assert_called_once()
    mock_model.get_classification.assert_called_once_with("/tmp/file.wav")
    assert output == {"bird": "species"}


@pytest.mark.usefixtures("inference_container")
@patch("app.inference.main.write_file_to_minio")
def test_save_classification_to_minio(mock_write_file_to_minio, mock_minio_client):
    output = {"bird": {"species": "classification"}}
    json_file_name = save_classification_to_minio(
        mock_minio_client, "bucket", "file.wav", output
    )
    mock_write_file_to_minio.assert_called_once()
    assert json_file_name == "file.json"


@pytest.mark.usefixtures("inference_container")
@patch("app.inference.main.publish_message")
def test_publish_feedback_message(mock_publish_message, mock_rabbitmq_channel):
    publish_feedback_message(
        mock_rabbitmq_channel,
        "feedback_queue",
        "file.wav",
        "file.json",
        "email@example.com",
        "12345",
    )
    mock_publish_message.assert_called_once()


@pytest.mark.usefixtures("inference_container")
@patch("app.inference.main.publish_feedback_message")
@patch("app.inference.main.save_classification_to_minio")
@patch("app.inference.main.perform_inference")
@patch("app.inference.main.download_wav_file")
def test_run_inference_pipeline(
    mock_download_wav_file,
    mock_perform_inference,
    mock_save_classification_to_minio,
    mock_publish_feedback_message,
):
    mock_perform_inference.return_value = {"bird": {"species": "classification"}}
    mock_save_classification_to_minio.return_value = "file.json"
    run_inference_pipeline("minio_path", "email@example.com", "12345")
    mock_download_wav_file.assert_called_once()
    mock_perform_inference.assert_called_once()
    mock_save_classification_to_minio.assert_called_once()
    mock_publish_feedback_message.assert_called_once()


@pytest.mark.usefixtures("inference_container")
def test_inference_pipeline(set_env_var, mock_minio_client, mock_rabbitmq_channel):
    # Mock MinIO client
    mock_minio_client.fput_object.return_value = None
    mock_minio_client.stat_object.return_value = MagicMock()

    # Mock RabbitMQ connection and channel
    mock_rabbitmq_channel.basic_publish.return_value = None

    # Send a message to the inference queue
    message = {
        "minio_path": "mock_audio.wav",
        "email": "test@example.com",
        "ticket_number": "12345",
    }
    mock_rabbitmq_channel.basic_publish(
        exchange="", routing_key="api-to-inf-queue", body=json.dumps(message)
    )

    # Wait for the inference to complete and check the output
    time.sleep(10)  # Adjust the sleep time as needed

    # Check if the output file exists in MinIO
    output_file_path = "path/to/output.json"
    found = mock_minio_client.stat_object("minio-bucket-name", output_file_path)
    assert found is not None

    # Clean up
    mock_minio_client.remove_object("minio-bucket-name", "mock_audio.wav")
    mock_minio_client.remove_object("minio-bucket-name", output_file_path)
