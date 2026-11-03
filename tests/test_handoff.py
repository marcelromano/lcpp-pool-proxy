import asyncio
import json
import hashlib
import os
import threading

import pytest
from aiohttp import web

from conftest import Fake, Proxy, be, make_cfg
from pool_proxy.app import App
from pool_proxy.config import from_dict
from pool_proxy.router import RequestInfo


class SlotEngine(Fake):
    def __init__(self, name, directory):
        super().__init__(name)
        self.directory = directory
        self.events = []
        self.tokens = list(range(16))
        self.fail = ""
        self.prefilling = asyncio.Event()
        self.pause = None
        self.gguf = "/models/27b.gguf"
        self.np = 1
        self.cache_n = None
        self.engine_epoch = "fake-epoch"
        self.engine_id = "engine-" + name
        self.last_acked = 0
        self.operations = {}

    async def _chat(self, request):
        if self.fail == "decode":
            return web.json_response({"error": "failed"}, status=500)
        if self.fail == "decode_body":
            response = web.StreamResponse(headers={"Content-Length": "999"})
            await response.prepare(request)
            await response.write(b"truncated")
            request.transport.close()
            return response
        if self.cache_n is not None:
            self.requests.append({"body": await request.json()})
            return web.json_response({"choices": [], "timings": {"cache_n": self.cache_n}})
        return await super()._chat(request)

    def extra_routes(self, app):
        app.router.add_get("/props", self.props)
        app.router.add_get("/slots", self.slots)
        app.router.add_post("/apply-template", self.template)
        app.router.add_post("/tokenize", self.tokenize)
        app.router.add_post("/completion", self.completion)
        app.router.add_post("/slots/{slot}", self.action)
        app.router.add_get("/slot-operations", self.capability)
        app.router.add_get("/slot-operations/{op_id}", self.operation)
        app.router.add_post("/slot-operations/{op_id}/ack", self.ack)

    async def capability(self, request):
        return web.json_response({"capability": "op_id-terminal-v1", "engine_epoch": self.engine_epoch,
                                  "engine_id": self.engine_id, "owner": "pool-proxy", "next_seq": self.last_acked + 1})

    async def operation(self, request):
        return web.json_response(self.operations.get(request.match_info["op_id"], {"error": "unknown"}),
                                 status=200 if request.match_info["op_id"] in self.operations else 404)

    async def ack(self, request):
        body = await request.json()
        if (body.get("engine_epoch") != self.engine_epoch or type(body.get("owner_seq")) is not int
                or body.get("owner") != "pool-proxy" or body.get("engine_id") != self.engine_id):
            return web.json_response({"error": "identity claims required"}, status=400)
        op_id = request.match_info["op_id"]
        op = self.operations.get(op_id)
        if not op:
            return web.Response(status=204)
        if op["status"] not in {"completed", "failed", "cancelled_before_start"}:
            return web.json_response({"error": "not terminal"}, status=409)
        if body["owner_seq"] != op["owner_seq"]:
            return web.json_response({"error": "sequence mismatch"}, status=409)
        self.last_acked = max(self.last_acked, op["owner_seq"])
        del self.operations[op_id]
        return web.Response(status=204)

    async def props(self, request):
        return web.json_response({"model_path": self.gguf})

    async def slots(self, request):
        return web.json_response([{"id": i, "is_processing": self.fail == "busy", "has_media": False}
                                  for i in range(self.np)])

    async def template(self, request):
        self.events.append(("template", await request.json()))
        return web.json_response({"prompt": "templated"})

    async def tokenize(self, request):
        return web.json_response({"tokens": self.tokens})

    async def completion(self, request):
        body = await request.json()
        self.events.append(("prefill", body))
        self.prefilled = body["prompt"]
        self.prefilling.set()
        if self.pause:
            await self.pause.wait()
        return web.json_response({"content": ""}, status=500 if self.fail == "prefill" else 200)

    async def action(self, request):
        op = request.query["action"]
        b = await request.json()
        if op in ("save", "restore", "erase"):
            required = {"op_id", "engine_epoch", "owner_seq", "owner", "engine_id"}
            if op in ("save", "restore"):
                required.add("filename")
            if set(b) != required or b.get("owner") != "pool-proxy" or b.get("engine_id") != self.engine_id:
                return web.json_response({"error": "wire shape or identity mismatch"}, status=409)
            op_id = b.get("op_id")
            if not op_id:
                return web.json_response({"error": "op id required"}, status=409)
            if op_id in self.operations:
                return web.json_response(self.operations[op_id])
            if (b.get("engine_epoch") != self.engine_epoch or b.get("owner_seq") != self.last_acked + 1):
                return web.json_response({"error": "stale sequence or epoch"}, status=409)
            if self.fail == op:
                envelope = {"op_id": op_id, "engine_id": self.engine_id, "engine_epoch": self.engine_epoch,
                            "owner": "pool-proxy", "owner_seq": b["owner_seq"], "action": op,
                            "slot": int(request.match_info["slot"]), "filename": b.get("filename", ""),
                            "status": "failed", "error": {"message": "injected"}}
                self.operations[op_id] = envelope
                return web.json_response(envelope)
            result = await self.apply_action(op, b)
            envelope = {"op_id": op_id, "engine_id": self.engine_id, "engine_epoch": self.engine_epoch,
                        "owner": "pool-proxy", "owner_seq": b["owner_seq"], "action": op,
                        "slot": int(request.match_info["slot"]), "filename": b.get("filename", ""),
                        "status": "completed", "result": result}
            self.operations[op_id] = envelope
            return web.json_response(envelope)
        return web.json_response(await self.apply_action(op, b))

    async def apply_action(self, op, b):
        self.events.append((op, b))
        if op == "erase":
            return {"n_erased": 1}
        path = self.directory / b["filename"]
        if op == "save":
            path.write_bytes(b"state")
            for suffix in (".dft", ".ckpt"):
                (self.directory / (b["filename"] + suffix)).write_bytes(suffix.encode())
            return {"n_saved": len(self.prefilled), "n_written": 5}
        assert path.read_bytes() == b"state"
        assert (self.directory / (b["filename"] + ".dft")).read_bytes() == b".dft"
        assert (self.directory / (b["filename"] + ".ckpt")).read_bytes() == b".ckpt"
        return {"n_restored": len(self.tokens) - 1 - (self.fail == "partial")}


@pytest.fixture
async def setup_pair(tmp_path):
    live = []

    async def make(transport="shared", enabled=True, np=1):
        src = tmp_path / "source"
        src.mkdir(exist_ok=True)
        dst = src if transport == "shared" else tmp_path / "target"
        dst.mkdir(exist_ok=True)
        gguf = tmp_path / "27b.gguf"
        gguf.write_bytes(b"same model")
        a, b = await SlotEngine("3090", src).start(), await SlotEngine("mi50", dst).start()
        a.np = b.np = np
        backends = [be(f, np=np, prefill_tps=1000 if f is a else 100,
                       handoff_model_path=f.gguf, handoff_gguf_path=str(gguf),
                       handoff_state_format="mainline-q8-q4") for f in (a, b)]
        cfg = make_cfg(backends, probe_enabled=False, long_prompt_tokens=1,
                       handoff_enabled=enabled, handoff_source="3090", handoff_target="mi50",
                       handoff_models=["pool"], handoff_min_tokens=8,
                       handoff_transport=transport, handoff_source_dir=str(src), handoff_target_dir=str(dst))
        p = await Proxy(App(cfg)).start()
        live.append((p, a, b))
        if enabled:
            for _ in range(200):
                if p.app.cold_handoff.ready_digest(str(gguf)) is not None:
                    break
                await asyncio.sleep(0.005)
            else:
                raise AssertionError("startup hash did not finish")
        return p, a, b

    yield make
    for p, a, b in live:
        await p.stop()
        await a.stop()
        await b.stop()


U = [{"role": "user", "content": "a" * 60}]


@pytest.mark.parametrize("transport", ["shared", "copy"])
@pytest.mark.parametrize("stream", [False, True])
async def test_handoff_success(setup_pair, transport, stream):
    p, a, b = await setup_pair(transport)
    status, backend, reason, text = await p.chat(U, body={"stream": stream, "temperature": 0.3})
    assert (status, backend, reason) == (200, "mi50", "handoff")
    assert [e[0] for e in a.events] == ["template", "prefill", "save"]
    assert [e[0] for e in b.events] == ["template", "restore"]
    prefill = a.events[1][1]
    assert prefill == {"prompt": list(range(15)), "n_predict": 0, "cache_prompt": True, "id_slot": 0}
    assert b.requests[0]["body"] == {"model": "pool", "messages": U, "stream": stream,
                                       "temperature": 0.3, "id_slot": 0, "cache_prompt": True}
    assert not list(a.directory.iterdir()) and not list(b.directory.iterdir())
    assert all(st.inflight == 0 and not st.handoff_busy for st in p.app.pool.states.values())
    assert p.app.pool._affinity_lookup(RequestInfo.from_body({"messages": U + U}, p.app.cfg.settings)) == "mi50"


@pytest.mark.parametrize("stage", ["prefill", "save", "restore", "partial", "busy", "gguf", "tokens", "hash"])
async def test_failures_fall_back(setup_pair, tmp_path, stage):
    p, a, b = await setup_pair("copy")
    if stage in ("prefill", "save"):
        a.fail = stage
    elif stage == "gguf":
        b.gguf = "/models/35b.gguf"
    elif stage == "tokens":
        b.tokens[-1] = 999
    elif stage == "hash":
        other = tmp_path / "other.gguf"
        other.write_bytes(b"other quant")
        p.app.pool.states["mi50"].cfg.handoff_gguf_path = str(other)
        p.app.cold_handoff.ready_digest(str(other))
        await asyncio.gather(*list(p.app.cold_handoff.hash_tasks.values()))
    else:
        b.fail = stage
    status, backend, reason, _ = await p.chat(U)
    assert status == 200 and backend == "3090" and reason != "handoff"
    assert not b.requests
    assert not list(a.directory.iterdir()) and not list(b.directory.iterdir())
    assert all(st.inflight == 0 and not st.handoff_busy for st in p.app.pool.states.values())
    assert p.app.metrics.handoffs["fallback"] == 1


@pytest.mark.parametrize("kind", ["disabled", "short", "model", "slot", "image", "reuse", "mirror", "occupied"])
async def test_other_requests_keep_routing(setup_pair, kind):
    p, a, b = await setup_pair(enabled=kind != "disabled")
    body = {"model": "pool", "messages": U}
    if kind == "short":
        body["messages"] = [{"role": "user", "content": "short"}]
    if kind == "model":
        body["model"] = "other"
    if kind == "slot":
        body["id_slot"] = 0
    if kind == "image":
        body["messages"] = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]
    info = RequestInfo.from_body(body, p.app.cfg.settings)
    if kind == "reuse":
        info.reuse["mi50"] = 1
    if kind == "mirror":
        info.chain.append("extension")
        p.app.pool.record_affinity(RequestInfo.from_body(body, p.app.cfg.settings), "mi50")
    if kind == "occupied":
        p.app.pool.states["3090"].queued = 1
    assert await p.app.cold_handoff.attempt(None, b"", body, info) is None
    assert not a.events and not b.events


async def test_exact_threshold(setup_pair):
    p, a, b = await setup_pair()
    a.tokens = b.tokens = [1, 2]
    assert (await p.chat(U))[1] == "3090"
    assert not any(e[0] == "prefill" for e in a.events)


async def test_normal_request_waits_and_source_is_released(setup_pair):
    p, a, b = await setup_pair()
    a.pause = asyncio.Event()
    first = asyncio.create_task(p.chat(U))
    await asyncio.wait_for(a.prefilling.wait(), 2)
    second = asyncio.create_task(p.chat([{"role": "user", "content": "short"}]))
    await asyncio.sleep(0.05)
    assert not a.requests and not b.requests
    a.pause.set()
    results = await asyncio.gather(first, second)
    assert all(r[0] == 200 for r in results)
    assert results[0][1:3] == ("mi50", "handoff")
    assert all(st.inflight == 0 and st.queued == 0 for st in p.app.pool.states.values())


async def test_timeout_releases_reservations(setup_pair):
    p, a, b = await setup_pair()
    a.pause = asyncio.Event()
    p.app.cfg.settings.handoff_timeout_s = 0.05
    assert (await p.chat(U))[1] == "3090"
    assert all(st.inflight == 0 and not st.handoff_busy for st in p.app.pool.states.values())


async def test_cancellation_releases_reservations(setup_pair):
    p, a, b = await setup_pair()
    a.pause = asyncio.Event()
    body = {"model": "pool", "messages": U}
    info = RequestInfo.from_body(body, p.app.cfg.settings)
    task = asyncio.create_task(p.app.cold_handoff.attempt(None, b"", body, info))
    await asyncio.wait_for(a.prefilling.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert all(st.inflight == 0 and st.queued == 0 and not st.handoff_busy for st in p.app.pool.states.values())


def test_disabled_default_and_enabled_validation():
    assert not from_dict({"backends": [{"name": "a", "url": "http://localhost"}]}).settings.handoff_enabled
    with pytest.raises(ValueError, match="source/target"):
        from_dict({"backends": [{"name": "a", "url": "http://localhost"}], "settings": {"handoff_enabled": True}})


@pytest.mark.parametrize("fail", ["decode", "decode_body", "cache"])
async def test_decode_failure_before_delivery_falls_back(setup_pair, fail):
    p, a, b = await setup_pair()
    if fail.startswith("decode"):
        b.fail = fail
    else:
        b.cache_n = 0
    assert (await p.chat(U))[1] == "3090"
    assert p.app.metrics.handoffs["fallback"] == 1


async def test_source_available_during_decode_and_np2_reserved(setup_pair):
    p, a, b = await setup_pair(np=2)
    b.delay = 0.2
    task = asyncio.create_task(p.chat(U))
    for _ in range(100):
        if b.active:
            break
        await asyncio.sleep(0.005)
    assert b.active == 1
    assert p.app.pool.states["3090"].inflight == 0
    assert p.app.pool.states["mi50"].inflight == 2
    assert (await p.chat(U))[1] == "3090"
    assert (await task)[1] == "mi50"
    assert all(st.inflight == 0 for st in p.app.pool.states.values())


async def test_pending_hash_routes_without_reserving_or_blocking(setup_pair, monkeypatch):
    p, a, b = await setup_pair()
    cold = p.app.cold_handoff
    cold.hashes.clear()
    started, release = threading.Event(), threading.Event()
    original = cold.digest
    calls = []

    def slow(path):
        calls.append(path)
        started.set()
        assert release.wait(5)
        return original(path)

    monkeypatch.setattr(cold, "digest", slow)
    try:
        for _ in range(3):
            result = await asyncio.wait_for(p.chat(U), 1)
            assert result[0:3] == (200, "3090", "long_cold")
        assert started.is_set() and len(calls) == 1
        assert not a.events and not b.events
        assert p.app.metrics.handoffs["hash_not_ready"] == 3
        assert all(st.inflight == 0 and not st.handoff_busy for st in p.app.pool.states.values())
    finally:
        release.set()
        await asyncio.gather(*list(cold.hash_tasks.values()))
    assert (await p.chat(U))[1:3] == ("mi50", "handoff")
    assert len(calls) == 1


@pytest.mark.parametrize("change", ["size", "mtime", "path"])
async def test_hash_cache_invalidates(setup_pair, tmp_path, change):
    p, a, b = await setup_pair()
    cold = p.app.cold_handoff
    path = tmp_path / "27b.gguf"
    old_key = cold.hash_key(path)
    old = cold.ready_digest(path)
    assert old == hashlib.sha256(b"same model").hexdigest()
    if change == "size":
        path.write_bytes(b"different model, different size")
    elif change == "mtime":
        path.write_bytes(b"new! model")
        os.utime(path, ns=(old_key[2], old_key[2] + 1_000_000_000))
    else:
        path = tmp_path / "replacement.gguf"
        path.write_bytes(b"new model")
    assert cold.ready_digest(path) is None
    if change != "path":
        assert old_key not in cold.hashes
    await asyncio.gather(*list(cold.hash_tasks.values()))
    assert cold.ready_digest(path) == hashlib.sha256(path.read_bytes()).hexdigest()
    assert cold.ready_digest(path) != old


async def test_missing_hash_file_routes_normally_then_recovers(setup_pair, tmp_path):
    p, a, b = await setup_pair()
    path = tmp_path / "27b.gguf"
    path.unlink()
    assert (await p.chat(U))[1] == "3090"
    assert not a.events and not b.events
    path.write_bytes(b"same model")
    cold = p.app.cold_handoff
    assert cold.ready_digest(path) is None
    await asyncio.gather(*list(cold.hash_tasks.values()))
    assert cold.ready_digest(path) is not None


async def test_changed_file_during_background_hash_is_not_published(setup_pair, tmp_path, monkeypatch):
    p, a, b = await setup_pair()
    cold = p.app.cold_handoff
    cold.hashes.clear()
    original = cold.digest
    path = tmp_path / "27b.gguf"

    def changed(name):
        value = original(name)
        path.write_bytes(b"changed after read")
        return value

    monkeypatch.setattr(cold, "digest", changed)
    assert cold.ready_digest(path) is None
    await asyncio.gather(*list(cold.hash_tasks.values()))
    assert not cold.hashes
    monkeypatch.setattr(cold, "digest", original)
    assert cold.ready_digest(path) is None
    await asyncio.gather(*list(cold.hash_tasks.values()))
    assert cold.ready_digest(path) == hashlib.sha256(path.read_bytes()).hexdigest()
