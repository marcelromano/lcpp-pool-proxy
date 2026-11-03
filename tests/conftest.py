import asyncio
import os
from pathlib import Path
import tempfile

import aiohttp
import pytest
from aiohttp import web

from pool_proxy import config as cfgmod
from pool_proxy.app import App

# Logs de request vao para arquivo temporario; sem isso a suite escreve no caminho de producao.
os.environ["POOL_REQUEST_LOG"] = str(Path(tempfile.mkdtemp(prefix="pool-proxy-test-reqlog-")) / "request-log.jsonl")


class Fake:
    """llama-server falso: /health, /v1/chat/completions (json ou SSE), registra o que recebe."""

    def __init__(self, name, delay=0.0, chunks=3, health=200):
        self.name, self.delay, self.chunks, self.health_status = name, delay, chunks, health
        self.requests: list[dict] = []
        self.active = 0
        self.max_active = 0
        self.disconnected = asyncio.Event()
        self.runner = None
        self.url = ""

    async def start(self):
        app = web.Application()
        app.router.add_get("/health", self._health)
        app.router.add_post("/v1/chat/completions", self._chat)
        self.extra_routes(app)
        self.runner = web.AppRunner(app, handler_cancellation=True)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        return self

    def extra_routes(self, app):
        pass

    async def stop(self):
        await self.runner.cleanup()

    async def _health(self, request):
        return web.json_response({"status": "ok"}, status=self.health_status)

    async def _chat(self, request):
        body = await request.json()
        self.requests.append({"headers": dict(request.headers), "body": body})
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if body.get("stream"):
                resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
                await resp.prepare(request)
                for i in range(self.chunks):
                    await resp.write(f'data: {{"b":"{self.name}","i":{i}}}\n\n'.encode())
                    await asyncio.sleep(self.delay / self.chunks)
                await resp.write(b"data: [DONE]\n\n")
                return resp
            await asyncio.sleep(self.delay)
            return web.json_response({"backend": self.name, "choices": []})
        except asyncio.CancelledError:
            self.disconnected.set()
            raise
        finally:
            self.active -= 1


def make_cfg(backends, models=None, **settings):
    # Tests use an isolated persistent ledger sibling to the snapshot dirs.
    test_ledger = settings.get("slotops_ledger_dir")
    if not test_ledger and settings.get("handoff_source_dir"):
        test_ledger = str(Path(settings["handoff_source_dir"]).parent / "slotops-ledgers")
    if not test_ledger:
        test_ledger = tempfile.mkdtemp(prefix="pool-proxy-test-ledger-")
    if test_ledger:
        for backend in backends:
            backend.setdefault("slotops_ledger_dir", test_ledger)
    s = {"health_interval_s": 3600, "queue_timeout_s": 5, **settings}
    return cfgmod.from_dict({"backends": backends, "settings": s, "models": models or ["pool"]})


def be(fake, **kw):
    d = {"name": fake.name, "url": fake.url, "np": 1, "prefill_tps": 500, "decode_tps": 30,
         "engine_id": "engine-" + fake.name, "engine_owner": "pool-proxy"}
    d.update(kw)
    return d


class Proxy:
    def __init__(self, app: App):
        self.app = app
        self.runner = None
        self.url = ""
        self.http = None

    async def start(self):
        self.runner = web.AppRunner(self.app.build(), handler_cancellation=True)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
        self.http = aiohttp.ClientSession()
        return self

    def extra_routes(self, app):
        pass

    async def stop(self):
        await self.http.close()
        await self.runner.cleanup()

    async def chat(self, messages, **kw):
        body = {"model": "pool", "messages": messages, **kw.pop("body", {})}
        async with self.http.post(self.url + "/v1/chat/completions", json=body, **kw) as r:
            txt = await r.text()
            return r.status, r.headers.get("X-Pool-Backend"), r.headers.get("X-Pool-Reason"), txt


@pytest.fixture
async def env(tmp_path):
    """Fabrica: env(fakes, backends_cfg, **settings) -> Proxy; limpa tudo no final."""
    things = []

    async def make(fakes, backends, **settings):
        for f in fakes:
            if not f.runner:
                await f.start()
        configured = [be(f, **b) if isinstance(b, dict) else b for f, b in zip(fakes, backends)]
        for index, backend in enumerate(configured):
            backend.setdefault("slotops_ledger_dir", str(tmp_path / f"slotops-{index}"))
        cfg = make_cfg(configured, **settings)
        p = await Proxy(App(cfg)).start()
        things.append((fakes, p))
        return p

    yield make
    for fakes, p in things:
        await p.stop()
        for f in fakes:
            if f.runner:
                try:
                    await f.stop()
                except Exception:
                    pass
