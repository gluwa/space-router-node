from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SR_", env_file=".env")

    NODE_PORT: int = 9090
    COORDINATION_API_URL: str = "http://localhost:8000"

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


settings = Settings()
