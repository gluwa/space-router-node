import logging
import warnings

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SR_", env_file=".env")

    NODE_PORT: int = 9090
    COORDINATION_API_URL: str = "http://localhost:8000"

    # Max concurrent proxy connections (DoS protection)
    MAX_CONNECTIONS: int = 256

    # Bind address — restrict to specific interface if needed
    BIND_ADDRESS: str = "0.0.0.0"

    NODE_LABEL: str = ""
    NODE_REGION: str = ""
    NODE_TYPE: str = "residential"

    PUBLIC_IP: str = ""  # Auto-detected if empty

    # UPnP / NAT-PMP automatic port forwarding
    UPNP_ENABLED: bool = True
    UPNP_LEASE_DURATION: int = 3600  # seconds; 0 = permanent

    BUFFER_SIZE: int = 65536
    REQUEST_TIMEOUT: float = 30.0
    RELAY_TIMEOUT: float = 300.0

    LOG_LEVEL: str = "INFO"

    # TLS — auto-generates a self-signed cert if files don't exist
    TLS_CERT_PATH: str = "certs/node.crt"
    TLS_KEY_PATH: str = "certs/node.key"

    # mTLS — Gateway authentication
    MTLS_ENABLED: bool = False
    GATEWAY_CA_CERT_PATH: str = "certs/gateway-ca.crt"


settings = Settings()

# Warn if Coordination API uses plain HTTP in non-localhost scenarios
if not settings.COORDINATION_API_URL.startswith("https://"):
    if "localhost" not in settings.COORDINATION_API_URL and "127.0.0.1" not in settings.COORDINATION_API_URL:
        warnings.warn(
            f"COORDINATION_API_URL uses plain HTTP ({settings.COORDINATION_API_URL}). "
            "This exposes registration data to MITM attacks. Use HTTPS in production.",
            stacklevel=1,
        )


# Testing override — never set in production
import os
_ALLOW_LOOPBACK_TARGETS = os.environ.get("SR_ALLOW_LOOPBACK_TARGETS", "").lower() == "true"
