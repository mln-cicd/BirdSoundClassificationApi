from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient
from minio import Minio

from app.api.main import app  # import your FastAPI instance


@pytest.fixture()
def test_app():
    return TestClient(app)


# conftest.py


@pytest.fixture(scope="session", autouse=True)
def set_env_variables(monkeypatch):
    monkeypatch.setenv("MINIO_ENDPOINT", "minio.example.com")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "your-access-key-id")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "your-secret-access-key")
    monkeypatch.setenv("MINIO_BUCKET", "your-bucket-name")
    monkeypatch.setenv("RABBITMQ_HOST", "localhost")
    monkeypatch.setenv("RABBITMQ_PORT", "5672")
    monkeypatch.setenv("RABBITMQ_QUEUE_API2INF", "api2inf-queue")
    monkeypatch.setenv("RABBITMQ_QUEUE_INF2API", "inf2api-queue")


@pytest.fixture()
def rabbit_connection():
    with patch("pika.BlockingConnection") as mock_connection:
        mock_conn = Mock()
        mock_connection.return_value = mock_conn
        yield mock_conn


@pytest.fixture()
def rabbit_channel(rabbit_connection):
    mock_channel = Mock()
    rabbit_connection.channel.return_value = mock_channel
    return mock_channel


@pytest.fixture()
def minio_client():
    mock_minio_client = Mock(spec=Minio)
    return mock_minio_client
