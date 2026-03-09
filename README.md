# Space Router Home Node

A daemon that runs on residential machines and acts as a proxy exit point for the [Space Router](https://github.com/gluwa/space-router) network.

Traffic from AI agents flows through the Space Router Proxy Gateway to this Home Node, which forwards requests from a residential IP address.

## How it works

```
AI Agent → Proxy Gateway (cloud) → Home Node (your machine) → Target website
```

The Home Node:
- Registers with the Coordination API on startup
- Accepts TLS-encrypted proxy connections from the Proxy Gateway
- Forwards traffic to target servers from your residential IP
- Auto-configures your router via UPnP for port forwarding
- Deregisters on shutdown

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure
export SR_COORDINATION_API_URL=https://spacerouter-coordination-api.fly.dev
export SR_UPNP_ENABLED=true

# Run
python -m app.main
```

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `SR_COORDINATION_API_URL` | `http://localhost:8000` | Coordination API URL |
| `SR_NODE_PORT` | `9090` | Port for incoming proxy connections |
| `SR_NODE_LABEL` | `""` | Human-readable label for this node |
| `SR_NODE_REGION` | `""` | Geographic region (e.g. `us-west`, `KR`) |
| `SR_NODE_TYPE` | `residential` | Node type |
| `SR_PUBLIC_IP` | auto-detected | Public IP (auto-detected if empty) |
| `SR_UPNP_ENABLED` | `true` | Enable UPnP port forwarding |
| `SR_UPNP_LEASE_DURATION` | `3600` | UPnP lease duration in seconds |
| `SR_TLS_CERT_PATH` | `certs/node.crt` | TLS certificate path (auto-generated) |
| `SR_TLS_KEY_PATH` | `certs/node.key` | TLS key path (auto-generated) |
| `SR_BUFFER_SIZE` | `65536` | TCP relay buffer size |
| `SR_REQUEST_TIMEOUT` | `30.0` | Connection timeout in seconds |
| `SR_RELAY_TIMEOUT` | `300.0` | Bidirectional relay timeout in seconds |
| `SR_LOG_LEVEL` | `INFO` | Log level |

## macOS launchd service

Install as a system service that starts at boot:

```bash
sudo cp launchd/com.spacerouter.homenode.plist /Library/LaunchDaemons/
sudo launchctl load /Library/LaunchDaemons/com.spacerouter.homenode.plist
```

## Pre-built binaries

Cross-platform binaries (macOS ARM64/x64, Windows x64, Linux x64) are built automatically and published as [GitHub Releases](https://github.com/gluwa/space-router-node/releases).

## Running tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

## API contract

The Home Node communicates with two components:

**Coordination API** (registration):
- `POST /nodes` — register on startup
- `PATCH /nodes/{id}/status` — set status to `offline` on shutdown

**Proxy Gateway** (inbound proxy traffic):
- Accepts TLS TCP connections on `SR_NODE_PORT`
- Handles `CONNECT host:port` for HTTPS tunneling
- Handles `GET http://...` for HTTP forwarding
- Strips all `X-SpaceRouter-*` and `Proxy-Authorization` headers before forwarding to targets

See [component-contracts.md](https://github.com/gluwa/space-router/blob/main/docs/component-contracts.md) for full specifications.
