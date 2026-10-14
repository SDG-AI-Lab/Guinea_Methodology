"""
Azure helpers: Key Vault secret loading and a Blob-Storage-backed cache.

Design goals
------------
* Nothing here is mandatory. If the Azure SDKs are unavailable or no container is
  configured, the functions degrade gracefully to "local only", so the pipeline
  still runs on a laptop or in CI.
* Secrets are loaded into os.environ *once* at startup via load_secrets_into_env
  so the rest of the codebase keeps reading plain environment variables.
* The cache mirror lets a fresh container (Azure Container Apps / ACI / Functions)
  reuse layers a previous run already downloaded, instead of re-hitting the APIs.

AWS -> Azure mapping
--------------------
  S3 bucket        -> Blob Storage container
  Secrets Manager  -> Key Vault
  boto3            -> azure-storage-blob + azure-identity + azure-keyvault-secrets
"""

import os
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from azure.storage.blob import BlobServiceClient
    from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError, AzureError
    _HAS_BLOB = True
except Exception:  # SDK not installed (e.g. local dev / CI)
    _HAS_BLOB = False

try:
    from azure.identity import DefaultAzureCredential
    from azure.keyvault.secrets import SecretClient
    _HAS_KEYVAULT = True
except Exception:
    _HAS_KEYVAULT = False


# ---------------------------------------------------------------------------
# Local development hydration (.env + service-account key file)
# ---------------------------------------------------------------------------
def _hydrate_local_env() -> None:
    """
    Make local development painless:
    1. Load a local .env file (if python-dotenv is installed and .env exists).
    2. If GEE_SA_KEY is not set but GEE_SA_KEY_FILE points to a JSON file,
       read that file's contents into GEE_SA_KEY. This lets you keep the
       service-account JSON as a (gitignored) file locally, while Azure supplies
       the inline contents via Key Vault.

    Never overwrites values already set, so the Key Vault path wins in production.
    """
    try:
        from dotenv import load_dotenv, find_dotenv
        path = find_dotenv(usecwd=True)
        if path:
            load_dotenv(path, override=False)
            logger.info("Loaded local .env from %s", path)
    except Exception:
        pass  # python-dotenv optional; env vars may already be set

    if not os.environ.get("GEE_SA_KEY"):
        key_file = os.environ.get("GEE_SA_KEY_FILE")
        if key_file and os.path.exists(key_file):
            try:
                with open(key_file, "r", encoding="utf-8") as fh:
                    os.environ["GEE_SA_KEY"] = fh.read()
                logger.info("Loaded GEE service-account key from file.")
            except OSError as e:
                logger.warning("Could not read GEE_SA_KEY_FILE: %s", e)


# ---------------------------------------------------------------------------
# Key Vault
# ---------------------------------------------------------------------------
def load_secrets_into_env(vault_url: Optional[str] = None,
                          secret_name: Optional[str] = None) -> None:
    """
    Load local .env (dev), then pull a JSON secret from Azure Key Vault (prod) and
    export every key/value into os.environ (without overwriting values already
    set). Safe to call at import time.

    In Key Vault, store ONE secret (default name 'landuse-config') whose value is
    a flat JSON object, e.g.:
        {
          "OPENTOPO_API_KEY": "...",
          "GEE_SA_KEY": "{...service-account-json...}",
          "GEE_PROJECT_ID": "my-project",
          "GEE_SERVICE_ACCOUNT": "sa@my-project.iam.gserviceaccount.com",
          "AZURE_STORAGE_CONNECTION_STRING": "...",
          "AZURE_BLOB_CONTAINER": "landuse"
        }
    (Key Vault secret NAMES can't contain underscores, which is why we keep the
    real variable names inside a single JSON secret rather than one secret each.)

    Auth uses DefaultAzureCredential: locally it picks up `az login`; in Azure it
    uses the container's Managed Identity (no keys needed).
    """
    # Local dev: load .env and expand GEE_SA_KEY_FILE first.
    _hydrate_local_env()

    vault_url = vault_url or os.environ.get("AZURE_KEY_VAULT_URL")
    if not vault_url:
        logger.info("No AZURE_KEY_VAULT_URL set; using environment (and .env) as-is.")
        return
    if not _HAS_KEYVAULT:
        logger.warning("azure-keyvault-secrets/identity not installed; skipping Key Vault.")
        return

    secret_name = secret_name or os.environ.get("AZURE_KEYVAULT_SECRET_NAME", "landuse-config")
    try:
        client = SecretClient(vault_url=vault_url, credential=DefaultAzureCredential())
        payload = client.get_secret(secret_name).value or "{}"
        data = json.loads(payload)
        for k, v in data.items():
            os.environ.setdefault(k, v if isinstance(v, str) else json.dumps(v))
        logger.info("Loaded %d secrets from Key Vault secret '%s'.", len(data), secret_name)
    except Exception as e:
        logger.error("Could not load Key Vault secret '%s': %s", secret_name, e)


# ---------------------------------------------------------------------------
# Blob-Storage-backed cache
# ---------------------------------------------------------------------------
class BlobCache:
    """
    Thin wrapper turning {container}/{prefix}/{key} into local files.

    Auth: pass a connection string (simplest, works locally) OR an account URL
    (e.g. https://<acct>.blob.core.windows.net) which uses DefaultAzureCredential
    / Managed Identity in Azure.

    Typical use inside a fetcher:
        cache = BlobCache(container, prefix, conn_str=...)
        local = cache.local_path(key)
        if cache.exists(key):
            cache.download(key, local); return local
        ... download from source into `local` ...
        cache.upload(key, local)
    """

    def __init__(self, container: Optional[str], prefix: str = "landuse",
                 local_root: str = "/tmp/landuse/cache",
                 connection_string: Optional[str] = None,
                 account_url: Optional[str] = None):
        self.container = container
        self.prefix = prefix.strip("/")
        self.local_root = local_root
        os.makedirs(self.local_root, exist_ok=True)

        self._cc = None  # container client
        if container and _HAS_BLOB:
            try:
                if connection_string:
                    svc = BlobServiceClient.from_connection_string(connection_string)
                elif account_url and _HAS_KEYVAULT:
                    svc = BlobServiceClient(account_url, credential=DefaultAzureCredential())
                else:
                    svc = None
                if svc is not None:
                    self._cc = svc.get_container_client(container)
                    try:
                        self._cc.create_container()  # no-op if it already exists
                    except ResourceExistsError:
                        pass
                    except AzureError:
                        pass  # may lack create perms; assume it exists
            except Exception as e:
                logger.warning("Blob Storage unavailable (%s); using local cache only.", e)
                self._cc = None

    # -- path helpers ------------------------------------------------------
    def _blob_name(self, key: str) -> str:
        return f"{self.prefix}/{key.lstrip('/')}"

    def local_path(self, key: str) -> str:
        path = os.path.join(self.local_root, key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    # -- existence ---------------------------------------------------------
    def exists(self, key: str) -> bool:
        """True if the object exists locally or in Blob Storage."""
        if os.path.exists(self.local_path(key)):
            return True
        if not self._cc:
            return False
        try:
            return self._cc.get_blob_client(self._blob_name(key)).exists()
        except AzureError:
            return False

    # -- transfer ----------------------------------------------------------
    def download(self, key: str, dest: Optional[str] = None) -> Optional[str]:
        dest = dest or self.local_path(key)
        if os.path.exists(dest):
            return dest
        if not self._cc:
            return dest if os.path.exists(dest) else None
        try:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as f:
                self._cc.get_blob_client(self._blob_name(key)).download_blob().readinto(f)
            return dest
        except (AzureError, ResourceNotFoundError) as e:
            logger.warning("Blob download failed for %s: %s", key, e)
            return None

    def upload(self, key: str, src: str) -> bool:
        if not self._cc:
            return False
        try:
            with open(src, "rb") as f:
                self._cc.upload_blob(name=self._blob_name(key), data=f, overwrite=True)
            return True
        except AzureError as e:
            logger.warning("Blob upload failed for %s: %s", key, e)
            return False

    def list_keys(self, subprefix: str = "") -> list:
        """List keys (relative to prefix) under an optional subprefix."""
        if not self._cc:
            root = os.path.join(self.local_root, subprefix)
            out = []
            for dp, _, files in os.walk(root):
                for f in files:
                    out.append(os.path.relpath(os.path.join(dp, f), self.local_root))
            return out
        base = self._blob_name(subprefix)
        keys = []
        for blob in self._cc.list_blobs(name_starts_with=base):
            keys.append(blob.name[len(self.prefix) + 1:])
        return keys
