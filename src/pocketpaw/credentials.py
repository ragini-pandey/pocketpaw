"""Encrypted credential storage for PocketPaw.

Changes:
  - 2026-02-06: Initial implementation — Fernet encryption with machine-derived PBKDF2 key.

Stores API keys and tokens in ~/.pocketpaw/secrets.enc instead of plaintext config.json.
Encryption key derived from machine identity (hostname + MAC + username) so the encrypted
file only works on the same machine/user. Salt stored in ~/.pocketpaw/.salt.
"""

import base64
import json
import logging
import os
import platform
from functools import lru_cache
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.argon2 import Argon2id
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)

# Fields that are considered secrets and must be stored encrypted.
SECRET_FIELDS: frozenset[str] = frozenset(
    {
        "telegram_bot_token",
        "openai_api_key",
        "anthropic_api_key",
        "openai_compatible_api_key",
        "openrouter_api_key",
        "discord_bot_token",
        "slack_bot_token",
        "slack_app_token",
        "whatsapp_access_token",
        "whatsapp_verify_token",
        "tavily_api_key",
        "brave_search_api_key",
        "parallel_api_key",
        "elevenlabs_api_key",
        "google_api_key",
        "google_oauth_client_id",
        "google_oauth_client_secret",
        "spotify_client_id",
        "spotify_client_secret",
        "matrix_access_token",
        "matrix_password",
        "teams_app_id",
        "teams_app_password",
        "gchat_service_account_key",
        "sarvam_api_key",
        "litellm_api_key",
    }
)


def _ensure_permissions(path: Path, mode: int = 0o600) -> None:
    """Set strict file permissions (owner read/write only)."""
    if not path.exists():
        return
    try:
        path.chmod(mode)
    except OSError:
        # Windows doesn't support chmod the same way — skip silently
        pass


def _ensure_dir_permissions(path: Path) -> None:
    """Set strict directory permissions (owner rwx only)."""
    _ensure_permissions(path, mode=0o700)


VERSION_2_HEADER = b"PAW\x02"


class CredentialStore:
    """Encrypted credential store (2026 Edition).

    Supports:
      - v1: Fernet + PBKDF2 (legacy)
      - v2: AES-256-GCM + Argon2id (modern standard)

    Storage:
      - ~/.pocketpaw/secrets.enc  (Encrypted JSON)
      - ~/.pocketpaw/.salt        (16-byte random salt)
    """

    def __init__(self, config_dir: Path | None = None):
        if config_dir is None:
            config_dir = Path.home() / ".pocketpaw"
        self._config_dir = config_dir
        self._secrets_path = config_dir / "secrets.enc"
        self._salt_path = config_dir / ".salt"
        self._cache: dict[str, str] | None = None

    def _get_machine_id(self) -> str:
        """Return a persistent machine identifier.

        Tries (in order):
          1. /etc/machine-id  (Linux — systemd)
          2. /var/lib/dbus/machine-id  (Linux — older dbus)
          3. platform.node()  (hostname — fallback)

        uuid.getnode() is intentionally NOT used because it returns a
        random MAC on systems without a discoverable NIC, producing a
        different value on every process start.
        """
        for p in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
            try:
                mid = Path(p).read_text().strip()
                if mid:
                    return mid
            except OSError:
                continue
        return platform.node()

    def _get_macos_hardware_uuid(self) -> str:
        """Extract macOS hardware UUID or fallback to system identifier."""
        import subprocess

        # If we're obviously in a CI environment, use a stable constant
        if os.environ.get("GITHUB_ACTIONS") == "true" or os.environ.get("CI") == "true":
            return "CI_ENVIRONMENT_ID"

        if platform.system() != "Darwin":
            # Fallback for non-macOS: use platform.node() as a surrogate "uuid"
            return platform.node()

        try:
            # Command to extract the Hardware UUID on macOS
            cmd = (
                "ioreg -rd1 -c IOPlatformExpertDevice | "
                "awk '/IOPlatformUUID/ { print $3 }' | tr -d '\"'"
            )
            result = subprocess.check_output(cmd, shell=True, text=True).strip()
            return result
        except (subprocess.SubprocessError, Exception):
            return platform.node()  # Last resort fallback

    def _get_machine_identity(self) -> bytes:
        """Build a machine-bound identity string."""
        parts = [
            self._get_machine_id(),
            self._get_macos_hardware_uuid(),
        ]
        try:
            parts.append(os.getlogin())
        except OSError:
            # Headless / CI environments may not have a login name
            parts.append(os.environ.get("USER", os.environ.get("USERNAME", "pocketpaw")))
        return "|".join(parts).encode("utf-8")

    def _get_or_create_salt(self) -> bytes:
        """Load existing salt or generate a new 16-byte salt."""
        self._config_dir.mkdir(parents=True, exist_ok=True)
        _ensure_dir_permissions(self._config_dir)

        if self._salt_path.exists():
            salt = self._salt_path.read_bytes()
            if len(salt) >= 16:
                return salt[:16]

        salt = os.urandom(16)
        self._salt_path.write_bytes(salt)
        _ensure_permissions(self._salt_path)
        return salt

    def _derive_key(self) -> bytes:
        """Derive a legacy Fernet key from machine identity + salt via PBKDF2."""
        salt = self._get_or_create_salt()
        # Note: v1 identity doesn't include macOS hardware UUID (handled in _load)
        parts = [self._get_machine_id()]
        try:
            parts.append(os.getlogin())
        except OSError:
            parts.append(os.environ.get("USER", os.environ.get("USERNAME", "pocketpaw")))
        identity = "|".join(parts).encode("utf-8")

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480_000,
        )
        raw_key = kdf.derive(identity)
        return base64.urlsafe_b64encode(raw_key)

    def _derive_key_v2(self, salt: bytes) -> bytes:
        """Derive an AES-256 key via Argon2id (2026 standard)."""
        kdf = Argon2id(
            salt=salt,
            length=32,
            iterations=3,
            memory_cost=65536,
            lanes=4,
        )
        return kdf.derive(self._get_machine_identity())

    def _load(self) -> dict[str, str]:
        """Decrypt and load secrets from disk with auto-migration support."""
        if self._cache is not None:
            return self._cache

        if not self._secrets_path.exists():
            self._cache = {}
            return self._cache

        try:
            raw_data = self._secrets_path.read_bytes()
            if not raw_data:
                self._cache = {}
                return self._cache

            decrypted_json: str | None = None

            # Check for version 2 header
            if raw_data.startswith(VERSION_2_HEADER):
                # Format: [HEADER][NONCE][CIPHERTEXT]
                nonce = raw_data[len(VERSION_2_HEADER) : len(VERSION_2_HEADER) + 12]
                ciphertext = raw_data[len(VERSION_2_HEADER) + 12 :]
                aesgcm = AESGCM(self._derive_key_v2(self._get_or_create_salt()))
                decrypted = aesgcm.decrypt(nonce, ciphertext, None)
                decrypted_json = decrypted.decode("utf-8")

            # Fallback to legacy Fernet (v1)
            elif raw_data.startswith(b"gAAAA"):
                fernet = Fernet(self._derive_key())
                decrypted = fernet.decrypt(raw_data)
                decrypted_json = decrypted.decode("utf-8")

                # Auto-migrate to v2 format
                if decrypted_json:
                    logger.info("Auto-migrating secrets.enc to v2 security format.")
                    data = json.loads(decrypted_json)
                    self._save(data)
                    self._cache = data
                    return data

            if decrypted_json:
                self._cache = json.loads(decrypted_json)
            else:
                raise ValueError("Unknown or corrupted secrets format.")

        except (InvalidToken, json.JSONDecodeError, Exception) as exc:
            logger.warning(
                "Failed to decrypt secrets.enc (machine changed? format error?): %s. "
                "Starting with empty credential store.",
                exc,
            )
            self._cache = {}

        return self._cache

    def _save(self, data: dict[str, str]) -> None:
        """Encrypt and write secrets to disk using v2 (AES-256-GCM + Argon2id)."""
        self._config_dir.mkdir(parents=True, exist_ok=True)
        _ensure_dir_permissions(self._config_dir)

        salt = self._get_or_create_salt()
        nonce = os.urandom(12)
        aesgcm = AESGCM(self._derive_key_v2(salt))

        plaintext = json.dumps(data).encode("utf-8")
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)

        # Build v2 payload: [HEADER][NONCE][CIPHERTEXT]
        payload = VERSION_2_HEADER + nonce + ciphertext

        self._secrets_path.write_bytes(payload)
        _ensure_permissions(self._secrets_path)
        self._cache = data

    def get(self, name: str) -> str | None:
        """Get a secret by name. Returns None if not found."""
        data = self._load()
        return data.get(name)

    def set(self, name: str, value: str) -> None:
        """Store a secret."""
        data = self._load()
        data[name] = value
        self._save(data)

    def delete(self, name: str) -> None:
        """Remove a secret."""
        data = self._load()
        if name in data:
            del data[name]
            self._save(data)

    def get_all(self) -> dict[str, str]:
        """Get a copy of all stored secrets."""
        return dict(self._load())

    def clear_cache(self) -> None:
        """Force re-read from disk on next access."""
        self._cache = None


@lru_cache
def get_credential_store() -> CredentialStore:
    """Get the singleton CredentialStore instance."""
    return CredentialStore()
