import logging
from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

logger = logging.getLogger(__name__)


def get_blob_service_client(storage_account: str) -> BlobServiceClient:
    account_url = f"https://{storage_account}.blob.core.windows.net"
    credential = DefaultAzureCredential()
    return BlobServiceClient(account_url, credential=credential)


def download_blob_directory(
    storage_account: str,
    container_name: str,
    blob_prefix: str,
    local_dir: str,
) -> str:
    """Download all blobs under a prefix to a local directory."""
    client = get_blob_service_client(storage_account)
    container = client.get_container_client(container_name)

    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)

    blobs = list(container.list_blobs(name_starts_with=blob_prefix))
    if not blobs:
        raise FileNotFoundError(
            f"No blobs found under {container_name}/{blob_prefix}"
        )

    count = 0
    for blob in blobs:
        relative = blob.name[len(blob_prefix):].lstrip("/")
        if not relative:
            continue

        target = local_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)

        blob_client = container.get_blob_client(blob.name)
        with open(target, "wb") as f:
            f.write(blob_client.download_blob().readall())
        count += 1

    logger.info("Downloaded %d files from %s/%s -> %s", count, container_name, blob_prefix, local_dir)
    return str(local_path)


def upload_file_to_blob(
    storage_account: str,
    container_name: str,
    blob_name: str,
    local_path: str,
) -> str:
    """Upload a local file to blob storage. Returns the blob URL."""
    client = get_blob_service_client(storage_account)
    blob_client = client.get_blob_client(container=container_name, blob=blob_name)

    with open(local_path, "rb") as f:
        blob_client.upload_blob(f, overwrite=True)

    logger.info("Uploaded %s -> %s/%s", local_path, container_name, blob_name)
    return blob_client.url
