"""Node registration with the Coordination API.

Lifecycle:
  1. detect_public_ip()   — determine the machine's public IP
  2. register_node()      — POST /nodes to announce ourselves
  3. deregister_node()    — PATCH /nodes/{id}/status → offline on shutdown
"""

import logging

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)

# Services tried in order for IP detection
_IP_SERVICES = [
    ("https://httpbin.org/ip", "origin"),
    ("https://api.ipify.org?format=json", "ip"),
    ("https://ifconfig.me/ip", None),  # plain-text response
]


async def detect_public_ip(http_client: httpx.AsyncClient) -> str:
    """Detect the machine's public IP by querying external services.

    Tries up to three services; returns the first successful result.
    Raises ``RuntimeError`` if all fail.
    """
    for url, json_key in _IP_SERVICES:
        try:
            resp = await http_client.get(url, timeout=10.0)
            resp.raise_for_status()
            if json_key:
                ip = resp.json()[json_key]
            else:
                ip = resp.text.strip()
            if ip:
                logger.info("Detected public IP: %s (via %s)", ip, url)
                return ip
        except Exception as exc:
            logger.debug("IP detection failed via %s: %s", url, exc)

    raise RuntimeError("Failed to detect public IP from all services")


async def register_node(
    http_client: httpx.AsyncClient,
    settings: Settings,
    public_ip: str,
    *,
    upnp_endpoint: tuple[str, int] | None = None,
) -> str:
    """Register this node with the Coordination API.

    If *upnp_endpoint* is provided (``(external_ip, external_port)``),
    the ``endpoint_url`` uses the UPnP-mapped address and the residential
    *public_ip* is sent as metadata.  Otherwise falls back to the public
    IP with the configured port (requires manual port forwarding).

    Returns the ``node_id`` assigned by the API.
    Raises on failure — the caller should abort startup.
    """
    if upnp_endpoint:
        upnp_ip, upnp_port = upnp_endpoint
        endpoint_url = f"https://{upnp_ip}:{upnp_port}"
        connectivity_type = "upnp"
    else:
        endpoint_url = f"https://{public_ip}:{settings.NODE_PORT}"
        connectivity_type = "direct"

    payload = {
        "endpoint_url": endpoint_url,
        "public_ip": public_ip,
        "connectivity_type": connectivity_type,
        "node_type": settings.NODE_TYPE,
    }
    if settings.NODE_REGION:
        payload["region"] = settings.NODE_REGION
    if settings.NODE_LABEL:
        payload["label"] = settings.NODE_LABEL

    url = f"{settings.COORDINATION_API_URL}/nodes"
    logger.info(
        "Registering node at %s → endpoint=%s public_ip=%s connectivity=%s",
        url, endpoint_url, public_ip, connectivity_type,
    )

    resp = await http_client.post(url, json=payload, timeout=15.0)
    resp.raise_for_status()
    data = resp.json()
    node_id = data["id"]
    ip_type = data.get("ip_type", "unknown")
    ip_region = data.get("ip_region", "unknown")
    logger.info(
        "Registered as node %s (ip_type=%s, ip_region=%s)",
        node_id, ip_type, ip_region,
    )
    return node_id


async def deregister_node(
    http_client: httpx.AsyncClient,
    settings: Settings,
    node_id: str,
) -> None:
    """Set node status to offline. Best-effort — failures are logged, not raised."""
    url = f"{settings.COORDINATION_API_URL}/nodes/{node_id}/status"
    try:
        resp = await http_client.patch(url, json={"status": "offline"}, timeout=10.0)
        resp.raise_for_status()
        logger.info("Deregistered node %s (status → offline)", node_id)
    except Exception as exc:
        logger.warning("Failed to deregister node %s: %s", node_id, exc)
