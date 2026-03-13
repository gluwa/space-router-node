"""Node registration with the Coordination API.

Lifecycle:
  1. detect_public_ip()   — determine the machine's public IP
  2. classify_ip()        — determine IP type, region, and AS info
  3. register_node()      — POST /nodes to announce ourselves
  4. deregister_node()    — PATCH /nodes/{id}/status → offline on shutdown
"""

import logging
import os
from dataclasses import dataclass

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


# ---------------------------------------------------------------------------
# IP classification — determine type, region, and AS info
# ---------------------------------------------------------------------------

@dataclass
class IPClassification:
    """Result of classifying a public IP address."""
    ip_type: str        # "residential", "hosting", or "unknown"
    ip_region: str      # ISO country code (e.g. "KR", "US")
    as_type: str        # AS description (e.g. "AS4766 Korea Telecom")

    @staticmethod
    def unknown() -> "IPClassification":
        return IPClassification(ip_type="unknown", ip_region="unknown", as_type="unknown")


# Services tried in order for IP classification
_IP_CLASSIFICATION_SERVICES = [
    "http://ip-api.com/json/{ip}?fields=status,countryCode,as,hosting",
    "https://ipapi.co/{ip}/json/",
]


async def classify_ip(http_client: httpx.AsyncClient, ip: str) -> IPClassification:
    """Classify an IP address to determine its type, region, and AS info.

    Tries multiple services in order; returns the first successful result.
    Returns an ``IPClassification`` with ``unknown`` fields if all fail.
    """
    # Service 1: ip-api.com (free, no key needed, HTTP only)
    try:
        url = _IP_CLASSIFICATION_SERVICES[0].format(ip=ip)
        resp = await http_client.get(url, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "success":
            is_hosting = data.get("hosting", False)
            ip_type = "hosting" if is_hosting else "residential"
            ip_region = data.get("countryCode", "unknown")
            as_type = data.get("as", "unknown")
            logger.info(
                "IP classified: type=%s, region=%s, as=%s (via ip-api.com)",
                ip_type, ip_region, as_type,
            )
            return IPClassification(ip_type=ip_type, ip_region=ip_region, as_type=as_type)
    except Exception as exc:
        logger.debug("IP classification failed via ip-api.com: %s", exc)

    # Service 2: ipapi.co (free tier, HTTPS)
    try:
        url = _IP_CLASSIFICATION_SERVICES[1].format(ip=ip)
        resp = await http_client.get(url, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("error"):
            ip_region = data.get("country_code", "unknown")
            as_type = f"AS{data.get('asn', '?')} {data.get('org', 'unknown')}"
            # ipapi.co doesn't directly tell us hosting vs residential,
            # but we can infer from org type
            ip_type = "residential"  # default assumption
            logger.info(
                "IP classified: type=%s, region=%s, as=%s (via ipapi.co)",
                ip_type, ip_region, as_type,
            )
            return IPClassification(ip_type=ip_type, ip_region=ip_region, as_type=as_type)
    except Exception as exc:
        logger.debug("IP classification failed via ipapi.co: %s", exc)

    logger.warning("Failed to classify IP %s from all services", ip)
    return IPClassification.unknown()


async def register_node(
    http_client: httpx.AsyncClient,
    settings: Settings,
    public_ip: str,
    *,
    upnp_endpoint: tuple[str, int] | None = None,
    ip_classification: IPClassification | None = None,
) -> tuple[str, str | None]:
    """Register this node with the Coordination API.

    If *upnp_endpoint* is provided (``(external_ip, external_port)``),
    the ``endpoint_url`` uses the UPnP-mapped address and the residential
    *public_ip* is sent as metadata.  Otherwise falls back to the public
    IP with the configured port (requires manual port forwarding).

    If *ip_classification* is provided, includes ``ip_type``, ``ip_region``,
    and ``as_type`` in the registration payload so the coordination API can
    use client-side classification when server-side lookup is unavailable.

    Returns ``(node_id, gateway_ca_cert_pem_or_None)``.
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

    # Include client-side IP classification if available
    if ip_classification and ip_classification.ip_type != "unknown":
        payload["ip_type"] = ip_classification.ip_type
        payload["ip_region"] = ip_classification.ip_region
        payload["as_type"] = ip_classification.as_type

    url = f"{settings.COORDINATION_API_URL}/nodes"
    logger.info(
        "Registering node at %s → endpoint=%s public_ip=%s connectivity=%s",
        url, endpoint_url, public_ip, connectivity_type,
    )

    resp = await http_client.post(url, json=payload, timeout=15.0)
    resp.raise_for_status()
    data = resp.json()
    node_id = data["id"]
    gateway_ca_cert = data.get("gateway_ca_cert")
    # Prefer server-side classification; fall back to client-side
    ip_type = data.get("ip_type") or (ip_classification.ip_type if ip_classification else "unknown")
    ip_region = data.get("ip_region") or (ip_classification.ip_region if ip_classification else "unknown")
    as_type = data.get("as_type") or (ip_classification.as_type if ip_classification else "unknown")
    logger.info(
        "Registered as node %s (ip_type=%s, ip_region=%s, as=%s, mtls_ca=%s)",
        node_id, ip_type, ip_region, as_type,
        "provided" if gateway_ca_cert else "not provided",
    )
    return node_id, gateway_ca_cert


def save_gateway_ca_cert(pem_data: str, path: str) -> None:
    """Write the gateway CA certificate PEM to disk."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(pem_data)
    os.chmod(path, 0o644)
    logger.info("Gateway CA certificate saved to %s", path)


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
