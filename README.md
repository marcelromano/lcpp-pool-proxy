# lcpp-pool-proxy

An asynchronous HTTP pool proxy (aiohttp) that routes and admits OpenAI-style chat requests across several
`llama-server` (llama.cpp) backends, for example one RTX 3090 and two AMD MI50 cards. It exposes a single
`api_base` in front of heterogeneous engines.

Part of the code was developed with the help of AI agents.

## Architecture

- **Routing** (`router.py`): eligible backends (healthy, vision if the request has images, enough `ctx_max`),
  then prefix affinity, then cold long prompts go to the highest `prefill_tps`, otherwise lowest estimated cost
  (queue wait + prompt/prefill_tps + max_tokens/decode_tps, divided by `weight`).
- **Admission**: per-backend concurrency equals `np`; long waits spill to backends marked `overflow: true`.
- **Slot probe** (`probe.py`): optional prefix-reuse affinity from a llama-server `/slots` telemetry patch
  (`prefix_hash`); engines without it fall back to a local message-hash mirror.
- **Router mode and mode switching** (`routerctl.py`, `modes.py`): drives a llama-server router that hosts
  several models, switching the active mode on demand, with slot save/restore (`slotops.py`, `sessions.py`).
- **Prefill/decode handoff** (`handoff.py`, `swap35.py`): opt-in prefill on one backend and decode on another
  by transferring the KV state file. Requires matching GGUF identities; see `config.example.yaml`.
- **Tool schema repair** (`schemas.py`), **request log** (`request_log.py`), **Prometheus metrics** (`metrics.py`).

Endpoints: `/v1/chat/completions` (stream and non-stream), `/v1/models`, `/health`, `/metrics`.
Responses carry `X-Pool-Backend` and `X-Pool-Reason`. Client disconnects close the upstream connection.

## Configuration

Copy `config.example.yaml` to `config.yaml` and edit backend URLs and capacities. API keys are read only from
environment variables named by `key_env` in each backend (never logged). `deploy/` holds further example
configs (router mode, handoff, shadow mode) and a systemd unit (`deploy/pool-proxy.service`, expects an
environment file at `/etc/pool-proxy.env`). `POOL_PROXY_STATE_DIR` sets where state and logs are kept.

## Running

    python -m venv .venv
    .venv/bin/pip install -r requirements.txt
    cp config.example.yaml config.yaml
    export POOL_KEY_RTX3090=... POOL_KEY_MI50=...
    .venv/bin/python -m pool_proxy -c config.yaml

## Tests

    .venv/bin/pip install -r requirements-dev.txt
    .venv/bin/python -m pytest

The suite uses fake backends on loopback; no GPU or llama-server is needed.

## Dependencies and external origins

Python 3.10+, aiohttp (Apache-2.0), PyYAML (MIT); tests use pytest and pytest-asyncio (MIT). It talks to the
llama.cpp server (MIT, https://github.com/ggml-org/llama.cpp). The slot telemetry and slot-operation endpoints
are optional features of patched llama-server builds, not part of this repository.

## License

MIT, see `LICENSE`.
