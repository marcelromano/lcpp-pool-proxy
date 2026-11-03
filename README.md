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

## Results

Measured on one host (X99 board, 2x Xeon E5-2673 v3, PCIe 3.0) with one RTX 3090 engine (Qwen3.8-27B, IQ3_S-based custom GGUF mix, 1 slot) and a two-MI50 llama-server router (Qwen3.8-27B and Qwen3.6-35B-A3B IQ4_XS, 2 slots). Test suite: 198 tests pass in 83 s (Python 3.14, aiohttp) on a clean checkout.

**Overhead per hop** (30 interleaved sequential calls per route, 1 prompt token and 1 output token, same backend and model, wall time at the client):

| Route | p50 | p95 | Cost of the hop (paired, p50 / p95) |
|---|---:|---:|---|
| Engine directly | 128.6 ms | 145.6 ms | |
| Through the pool proxy | 130.0 ms | 136.2 ms | +0.7 ms / +7.3 ms |
| Through an ASGI gateway in front of the proxy | 152.7 ms | 205.3 ms | +22.7 ms / +74.4 ms over the proxy |

Single-token sequential load only; streaming and concurrent load were not measured.

**Prefill/decode handoff for Qwen3.6-35B-A3B (IQ4_XS)** (prefill on the RTX 3090 while it is borrowed from the 27B, decode on the MI50 cards, KV transferred as a state file):

| Prompt | Direct on 2x MI50 | With handoff | Speedup |
|---:|---:|---:|---:|
| 79,089 tokens (TTFT) | 191.7 s / 192.3 s | 53.3 s | 3.6x (-72%) |
| 127,685 tokens (TTFT) | not measured | 88.6 s | |

Cost of the swap window: 7.1-9.2 s to open and 6.6-9.6 s to close (27B restored). In a 30-minute soak with 55 serial requests plus 63 parallel 27B requests there were 0 non-200 responses; 9 handoffs had a TTFT of 51.0-52.9 s; the 27B stayed at p50 0.64 s serial and 0.37 s parallel. A full day logged 24 openings, 24 closings, 24 restores. When the prefill engine was killed mid-prefill, the client still got a 200 from the 35B on the MI50 cards and the 27B was back in 6.7 s.

**Single entry point.** After routing all local models through the proxy, a 10-minute soak (67 requests, 4 engines health-checked) gave 0 non-200 responses; a 30-minute soak on the MI50 router (304 requests, 158k prompt tokens, 0 errors) ran alongside the cancellation fix. A purge bug that expired a request waiting for a mode switch (503 after 180 s) is fixed; the same scenario now returns 200 in 141 s.

## What it enables

* One `api_base` in front of mixed engines: an RTX 3090 and a pair of MI50 cards behind the same OpenAI-style endpoint, with per-backend concurrency equal to the slot count and overflow spill.
* Cost-based routing (queue wait + prompt / prefill speed + output / decode speed) with prefix affinity, so repeat prompts land on the slot that already holds their KV.
* Hosting several models on a llama-server router with on-demand mode switching and slot save/restore.
* Long-prompt handoff between engines with a guaranteed fall-back to direct serving and mandatory restoration of the borrowed card.
* Negligible routing cost: about 1 ms at the median, below the noise of the engine itself.

## Dependencies and external origins

Python 3.10+, aiohttp (Apache-2.0), PyYAML (MIT); tests use pytest and pytest-asyncio (MIT). It talks to the
llama.cpp server (MIT, https://github.com/ggml-org/llama.cpp). The slot telemetry and slot-operation endpoints
are optional features of patched llama-server builds, not part of this repository.

## License

MIT, see `LICENSE`.
