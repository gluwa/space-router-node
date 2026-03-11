"""Tests for node registration and IP detection."""

import pytest
import respx
from httpx import Response

from app.config import Settings
from app.registration import deregister_node, detect_public_ip, register_node, save_gateway_ca_cert


@pytest.fixture
def reg_settings():
    return Settings(
        NODE_PORT=9090,
        COORDINATION_API_URL="http://coordination:8000",
        NODE_LABEL="test-node",
        NODE_REGION="us-west",
        NODE_TYPE="residential",
        PUBLIC_IP="",
    )


# ---------------------------------------------------------------------------
# detect_public_ip
# ---------------------------------------------------------------------------

class TestDetectPublicIP:
    @pytest.mark.asyncio
    @respx.mock
    async def test_first_service_succeeds(self):
        respx.get("https://httpbin.org/ip").mock(
            return_value=Response(200, json={"origin": "1.2.3.4"})
        )

        import httpx
        async with httpx.AsyncClient() as client:
            ip = await detect_public_ip(client)
        assert ip == "1.2.3.4"

    @pytest.mark.asyncio
    @respx.mock
    async def test_fallback_to_second_service(self):
        respx.get("https://httpbin.org/ip").mock(
            return_value=Response(500)
        )
        respx.get("https://api.ipify.org?format=json").mock(
            return_value=Response(200, json={"ip": "5.6.7.8"})
        )

        import httpx
        async with httpx.AsyncClient() as client:
            ip = await detect_public_ip(client)
        assert ip == "5.6.7.8"

    @pytest.mark.asyncio
    @respx.mock
    async def test_fallback_to_third_service(self):
        respx.get("https://httpbin.org/ip").mock(
            return_value=Response(500)
        )
        respx.get("https://api.ipify.org?format=json").mock(
            return_value=Response(500)
        )
        respx.get("https://ifconfig.me/ip").mock(
            return_value=Response(200, text="9.10.11.12")
        )

        import httpx
        async with httpx.AsyncClient() as client:
            ip = await detect_public_ip(client)
        assert ip == "9.10.11.12"

    @pytest.mark.asyncio
    @respx.mock
    async def test_all_services_fail(self):
        respx.get("https://httpbin.org/ip").mock(
            return_value=Response(500)
        )
        respx.get("https://api.ipify.org?format=json").mock(
            return_value=Response(500)
        )
        respx.get("https://ifconfig.me/ip").mock(
            return_value=Response(500)
        )

        import httpx
        async with httpx.AsyncClient() as client:
            with pytest.raises(RuntimeError, match="Failed to detect"):
                await detect_public_ip(client)


# ---------------------------------------------------------------------------
# register_node
# ---------------------------------------------------------------------------

class TestRegisterNode:
    @pytest.mark.asyncio
    @respx.mock
    async def test_register_success(self, reg_settings):
        respx.post("http://coordination:8000/nodes").mock(
            return_value=Response(201, json={
                "id": "node-abc-123",
                "endpoint_url": "http://1.2.3.4:9090",
                "node_type": "residential",
                "status": "online",
                "health_score": 1.0,
                "region": "us-west",
                "label": "test-node",
                "created_at": "2026-01-01T00:00:00Z",
            })
        )

        import httpx
        async with httpx.AsyncClient() as client:
            node_id, gateway_ca_cert = await register_node(client, reg_settings, "1.2.3.4")

        assert node_id == "node-abc-123"
        assert gateway_ca_cert is None

        # Verify the request payload
        req = respx.calls[0].request
        import json
        body = json.loads(req.content)
        assert body["endpoint_url"] == "https://1.2.3.4:9090"
        assert body["public_ip"] == "1.2.3.4"
        assert body["connectivity_type"] == "direct"
        assert body["node_type"] == "residential"
        assert body["region"] == "us-west"
        assert body["label"] == "test-node"

    @pytest.mark.asyncio
    @respx.mock
    async def test_register_with_upnp_endpoint(self, reg_settings):
        respx.post("http://coordination:8000/nodes").mock(
            return_value=Response(201, json={
                "id": "node-upnp-456",
                "endpoint_url": "https://203.0.113.5:9090",
                "public_ip": "1.2.3.4",
                "connectivity_type": "upnp",
                "node_type": "residential",
                "status": "online",
                "health_score": 1.0,
                "region": "us-west",
                "label": "test-node",
                "created_at": "2026-01-01T00:00:00Z",
            })
        )

        import httpx
        async with httpx.AsyncClient() as client:
            node_id, gateway_ca_cert = await register_node(
                client, reg_settings, "1.2.3.4",
                upnp_endpoint=("203.0.113.5", 9090),
            )

        assert node_id == "node-upnp-456"
        assert gateway_ca_cert is None

        req = respx.calls[0].request
        import json
        body = json.loads(req.content)
        assert body["endpoint_url"] == "https://203.0.113.5:9090"
        assert body["public_ip"] == "1.2.3.4"
        assert body["connectivity_type"] == "upnp"

    @pytest.mark.asyncio
    @respx.mock
    async def test_register_receives_ip_classification(self, reg_settings):
        """Registration response with ip_type/ip_region should be parsed without error."""
        respx.post("http://coordination:8000/nodes").mock(
            return_value=Response(201, json={
                "id": "node-classified",
                "endpoint_url": "https://1.2.3.4:9090",
                "public_ip": "1.2.3.4",
                "connectivity_type": "direct",
                "node_type": "residential",
                "status": "online",
                "health_score": 1.0,
                "region": "us-west",
                "label": "test-node",
                "ip_type": "residential",
                "ip_region": "Portland, US",
                "created_at": "2026-01-01T00:00:00Z",
            })
        )

        import httpx
        async with httpx.AsyncClient() as client:
            node_id, gateway_ca_cert = await register_node(client, reg_settings, "1.2.3.4")

        assert node_id == "node-classified"
        assert gateway_ca_cert is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_register_handles_missing_ip_classification(self, reg_settings):
        """Registration response without ip_type/ip_region should default to 'unknown'."""
        respx.post("http://coordination:8000/nodes").mock(
            return_value=Response(201, json={
                "id": "node-no-class",
                "endpoint_url": "https://1.2.3.4:9090",
                "node_type": "residential",
                "status": "online",
                "health_score": 1.0,
                "created_at": "2026-01-01T00:00:00Z",
                # No ip_type or ip_region — code uses .get() with "unknown" default
            })
        )

        import httpx
        async with httpx.AsyncClient() as client:
            node_id, gateway_ca_cert = await register_node(client, reg_settings, "1.2.3.4")

        # Should succeed without KeyError
        assert node_id == "node-no-class"
        assert gateway_ca_cert is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_register_failure_raises(self, reg_settings):
        respx.post("http://coordination:8000/nodes").mock(
            return_value=Response(500, text="Internal Server Error")
        )

        import httpx
        async with httpx.AsyncClient() as client:
            with pytest.raises(httpx.HTTPStatusError):
                await register_node(client, reg_settings, "1.2.3.4")

    @pytest.mark.asyncio
    @respx.mock
    async def test_register_returns_gateway_ca_cert(self, reg_settings):
        """Registration response with gateway_ca_cert should return it."""
        ca_pem = "-----BEGIN CERTIFICATE-----\nTESTDATA\n-----END CERTIFICATE-----"
        respx.post("http://coordination:8000/nodes").mock(
            return_value=Response(201, json={
                "id": "node-mtls-1",
                "endpoint_url": "https://1.2.3.4:9090",
                "node_type": "residential",
                "status": "online",
                "health_score": 1.0,
                "gateway_ca_cert": ca_pem,
                "created_at": "2026-01-01T00:00:00Z",
            })
        )

        import httpx
        async with httpx.AsyncClient() as client:
            node_id, gateway_ca_cert = await register_node(client, reg_settings, "1.2.3.4")

        assert node_id == "node-mtls-1"
        assert gateway_ca_cert == ca_pem


# ---------------------------------------------------------------------------
# save_gateway_ca_cert
# ---------------------------------------------------------------------------

class TestSaveGatewayCACert:
    def test_save_creates_file(self, tmp_path):
        ca_pem = "-----BEGIN CERTIFICATE-----\nTESTDATA\n-----END CERTIFICATE-----"
        path = str(tmp_path / "certs" / "gateway-ca.crt")
        save_gateway_ca_cert(ca_pem, path)

        with open(path) as f:
            assert f.read() == ca_pem

    def test_save_sets_permissions(self, tmp_path):
        import os
        import stat

        ca_pem = "-----BEGIN CERTIFICATE-----\nTESTDATA\n-----END CERTIFICATE-----"
        path = str(tmp_path / "gateway-ca.crt")
        save_gateway_ca_cert(ca_pem, path)

        mode = os.stat(path).st_mode & 0o777
        assert mode == 0o644


# ---------------------------------------------------------------------------
# deregister_node
# ---------------------------------------------------------------------------

class TestDeregisterNode:
    @pytest.mark.asyncio
    @respx.mock
    async def test_deregister_success(self, reg_settings):
        respx.patch("http://coordination:8000/nodes/node-abc-123/status").mock(
            return_value=Response(200, json={"ok": True})
        )

        import httpx
        async with httpx.AsyncClient() as client:
            # Should not raise
            await deregister_node(client, reg_settings, "node-abc-123")

        req = respx.calls[0].request
        import json
        body = json.loads(req.content)
        assert body["status"] == "offline"

    @pytest.mark.asyncio
    @respx.mock
    async def test_deregister_failure_logged_not_raised(self, reg_settings):
        respx.patch("http://coordination:8000/nodes/node-abc-123/status").mock(
            return_value=Response(500)
        )

        import httpx
        async with httpx.AsyncClient() as client:
            # Should NOT raise — deregister is best-effort
            await deregister_node(client, reg_settings, "node-abc-123")
