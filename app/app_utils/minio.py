"""MinIO Utility Module.

This module provides utility functions for interacting with MinIO object storage. It
includes functions for ensuring bucket existence, writing files to MinIO, and fetching
files from MinIO.

"""

import io
import logging
import os

logging.basicConfig(level=logging.INFO)


def ensure_bucket_exists(minio_client, bucket_name) -> None:
    """Ensure that the specified bucket exists in MinIO, creating it if necessary.

    Args:
        minio_client (Minio): MinIO client instance.
        bucket_name (str): Name of the bucket to check or create.

    """
    logging.info("Checking if bucket '%s' exists...", bucket_name)
    if not minio_client.bucket_exists(bucket_name):
        logging.info("Bucket '%s' does not exist. Creating bucket...", bucket_name)
        minio_client.make_bucket(bucket_name)
        logging.info("Bucket '%s' created successfully.", bucket_name)
    else:
        logging.info("Bucket '%s' already exists.", bucket_name)


def write_file_to_minio(minio_client, bucket_name, file_name, data) -> None:
    """Write a file to MinIO.

    Args:
        minio_client (Minio): MinIO client instance.
        bucket_name (str): Name of the bucket to write the file to.
        file_name (str): Name of the file to be written.
        data (Union[bytes, IOBase]): File data as bytes or a file-like object.

    """
    logging.info("Writing file '%s' to MinIO bucket '%s'...", file_name, bucket_name)
    if isinstance(data, bytes):
        length = len(data)
        data = io.BytesIO(data)
    else:
        length = data.seek(0, os.SEEK_END)
        data.seek(0)

    try:
        minio_client.put_object(bucket_name, file_name, data, length=length)
        logging.info("File '%s' written to MinIO bucket '%s' successfully.", file_name, bucket_name)
    except Exception as e:
        logging.error("Error writing file '%s' to MinIO bucket '%s': %s", file_name, bucket_name, str(e))
        raise


def fetch_file_from_minio(minio_client, bucket_name, file_name, local_file_path) -> bool:
    """Fetch a file from MinIO and saves it locally.

    Args:
        minio_client (Minio): MinIO client instance.
        bucket_name (str): Name of the bucket to fetch the file from.
        file_name (str): Name of the file to be fetched.
        local_file_path (str): Local file path to save the fetched file.

    Returns:
        bool: True if the file was fetched successfully, False otherwise.

    """
    logging.info("Fetching file '%s' from MinIO bucket '%s'...", file_name, bucket_name)
    try:
        minio_client.fget_object(bucket_name, file_name, local_file_path)
        logging.info(
            "File '%s' fetched from MinIO bucket '%s' and saved to '%s' successfully.",
            file_name,
            bucket_name,
            local_file_path,
        )
        return True
    except Exception as e:
        logging.error("Error fetching file '%s' from MinIO bucket '%s': %s", file_name, bucket_name, str(e))
        return False
