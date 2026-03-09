"""Tests for UPnP/NAT-PMP port mapping."""

from unittest.mock import MagicMock, patch

import pytest

from app.upnp import (
    _get_local_ip,
    remove_upnp_mapping,
    renew_upnp_mapping,
    setup_upnp_mapping,
)


class TestGetLocalIp:
    def test_returns_ipv4_string(self):
        ip = _get_local_ip()
        assert "." in ip
        parts = ip.split(".")
        assert len(parts) == 4


class TestSetupUpnpMapping:
    @pytest.mark.asyncio
    async def test_success(self):
        mock_upnp = MagicMock()
        mock_upnp.discover.return_value = 1
        mock_upnp.externalipaddress.return_value = "203.0.113.5"
        mock_upnp.addportmapping.return_value = None

        mock_module = MagicMock()
        mock_module.UPnP.return_value = mock_upnp

        with (
            patch.dict("sys.modules", {"miniupnpc": mock_module}),
            patch("app.upnp._get_local_ip", return_value="192.168.1.100"),
        ):
            result = await setup_upnp_mapping(9090, lease_duration=3600)

        assert result == ("203.0.113.5", 9090)
        mock_upnp.addportmapping.assert_called_once_with(
            9090, "TCP", "192.168.1.100", 9090,
            "SpaceRouter Home Node (9090/TCP)", "", 3600,
        )

    @pytest.mark.asyncio
    async def test_no_devices_found(self):
        mock_upnp = MagicMock()
        mock_upnp.discover.return_value = 0

        mock_module = MagicMock()
        mock_module.UPnP.return_value = mock_upnp

        with (
            patch.dict("sys.modules", {"miniupnpc": mock_module}),
            patch("app.upnp._get_local_ip", return_value="192.168.1.100"),
        ):
            result = await setup_upnp_mapping(9090)

        assert result is None

    @pytest.mark.asyncio
    async def test_selectigd_fails(self):
        mock_upnp = MagicMock()
        mock_upnp.discover.return_value = 1
        mock_upnp.selectigd.side_effect = Exception("no IGD found")

        mock_module = MagicMock()
        mock_module.UPnP.return_value = mock_upnp

        with (
            patch.dict("sys.modules", {"miniupnpc": mock_module}),
            patch("app.upnp._get_local_ip", return_value="192.168.1.100"),
        ):
            result = await setup_upnp_mapping(9090)

        assert result is None

    @pytest.mark.asyncio
    async def test_no_external_ip(self):
        mock_upnp = MagicMock()
        mock_upnp.discover.return_value = 1
        mock_upnp.externalipaddress.return_value = ""

        mock_module = MagicMock()
        mock_module.UPnP.return_value = mock_upnp

        with (
            patch.dict("sys.modules", {"miniupnpc": mock_module}),
            patch("app.upnp._get_local_ip", return_value="192.168.1.100"),
        ):
            result = await setup_upnp_mapping(9090)

        assert result is None

    @pytest.mark.asyncio
    async def test_addportmapping_fails(self):
        mock_upnp = MagicMock()
        mock_upnp.discover.return_value = 1
        mock_upnp.externalipaddress.return_value = "203.0.113.5"
        mock_upnp.addportmapping.side_effect = Exception("port conflict")

        mock_module = MagicMock()
        mock_module.UPnP.return_value = mock_upnp

        with (
            patch.dict("sys.modules", {"miniupnpc": mock_module}),
            patch("app.upnp._get_local_ip", return_value="192.168.1.100"),
        ):
            result = await setup_upnp_mapping(9090)

        assert result is None


class TestRemoveUpnpMapping:
    @pytest.mark.asyncio
    async def test_success(self):
        mock_upnp = MagicMock()
        mock_upnp.discover.return_value = 1

        mock_module = MagicMock()
        mock_module.UPnP.return_value = mock_upnp

        with patch.dict("sys.modules", {"miniupnpc": mock_module}):
            await remove_upnp_mapping(9090)

        mock_upnp.deleteportmapping.assert_called_once_with(9090, "TCP")

    @pytest.mark.asyncio
    async def test_removal_failure_logged_not_raised(self):
        mock_upnp = MagicMock()
        mock_upnp.discover.return_value = 1
        mock_upnp.deleteportmapping.side_effect = Exception("not found")

        mock_module = MagicMock()
        mock_module.UPnP.return_value = mock_upnp

        with patch.dict("sys.modules", {"miniupnpc": mock_module}):
            # Should not raise
            await remove_upnp_mapping(9090)

    @pytest.mark.asyncio
    async def test_no_devices_skips_removal(self):
        mock_upnp = MagicMock()
        mock_upnp.discover.return_value = 0

        mock_module = MagicMock()
        mock_module.UPnP.return_value = mock_upnp

        with patch.dict("sys.modules", {"miniupnpc": mock_module}):
            await remove_upnp_mapping(9090)

        mock_upnp.deleteportmapping.assert_not_called()


class TestRenewUpnpMapping:
    @pytest.mark.asyncio
    async def test_success_returns_true(self):
        mock_upnp = MagicMock()
        mock_upnp.discover.return_value = 1
        mock_upnp.externalipaddress.return_value = "203.0.113.5"

        mock_module = MagicMock()
        mock_module.UPnP.return_value = mock_upnp

        with (
            patch.dict("sys.modules", {"miniupnpc": mock_module}),
            patch("app.upnp._get_local_ip", return_value="192.168.1.100"),
        ):
            result = await renew_upnp_mapping(9090, 9090, 3600)

        assert result is True

    @pytest.mark.asyncio
    async def test_failure_returns_false(self):
        mock_upnp = MagicMock()
        mock_upnp.discover.return_value = 0

        mock_module = MagicMock()
        mock_module.UPnP.return_value = mock_upnp

        with (
            patch.dict("sys.modules", {"miniupnpc": mock_module}),
            patch("app.upnp._get_local_ip", return_value="192.168.1.100"),
        ):
            result = await renew_upnp_mapping(9090, 9090, 3600)

        assert result is False
