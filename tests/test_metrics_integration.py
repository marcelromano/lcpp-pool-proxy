"""Real HTTP integration of metrics middleware/_forward, without operational startup.

Routing/retry policy is intentionally out of scope: KvCapacity is returned by
this harness, as a final response, so upstream refusal is observable directly.
"""
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest

from pool_proxy.app import App, KvCapacity
from conftest import make_cfg
from pool_proxy.metrics import Metrics

ROUTE = '/v1/chat/completions'
SECRET = 'private-prompt-never-a-label'
RAW = ('{"n":2,"messages":[{"content":"' + SECRET + '"}]}').encode()


@asynccontextmanager
async def harness(status=200, body=b'\x00opaque\xff', stream=False, enabled=True):
    started, finish, completed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    seen = []

    async def upstream(request):
        seen.append(await request.read())
        headers = {'Content-Type': 'text/event-stream' if stream else 'application/octet-stream',
                   'X-Upstream': 'exact', 'Date': 'Tue, 01 Jan 2030 00:00:00 GMT'}
        if not stream:
            return web.Response(status=status, body=body, headers=headers)
        response = web.StreamResponse(status=status, headers=headers)
        await response.prepare(request)
        await response.write(b'data: first\n\n')
        started.set()
        await finish.wait()
        await response.write(b'data: [DONE]\n\n')
        await response.write_eof()
        return response

    upstream_app = web.Application()
    upstream_app.router.add_post(ROUTE, upstream)
    async with TestServer(upstream_app) as server, aiohttp.ClientSession(auto_decompress=False) as session:
        proxy = App.__new__(App)
        proxy.metrics, proxy.session = Metrics(), session
        proxy.cfg = make_cfg([{'name':'fake','url':'http://127.0.0.1:1','np':1}])
        proxy._started = True
        if not enabled:
            proxy.metrics._safe = lambda fn, *args: None
        state = SimpleNamespace(cfg=SimpleNamespace(url=str(server.make_url('/')), name='fake', key_env=None))

        async def forward(request, raw, body, info):
            try:
                try:
                    response, _, _ = await proxy._forward(request, state, 'integration', raw)
                    return response
                except KvCapacity as exc:
                    return exc.resp
            finally:
                completed.set()

        async def unrelated(request):
            return web.Response(body=b'health')

        proxy._chat_low = forward

        async def chat(request):
            request.headers  # exercise the production chat parser/n observation
            return await proxy.chat(request)

        app = web.Application(middlewares=[proxy._request_metrics])
        app.router.add_post(ROUTE, chat)
        app.router.add_get('/health', unrelated)
        app.router.add_get(ROUTE, unrelated)
        async with TestClient(TestServer(app), headers={proxy.cfg.settings.low_priority_header: 'low'}) as client:
            try:
                yield SimpleNamespace(client=client, metrics=proxy.metrics, seen=seen,
                                      started=started, finish=finish, completed=completed)
            finally:
                finish.set()


@pytest.mark.asyncio
@pytest.mark.parametrize('status,body,refusals', [
    (200, b'\x00opaque\xff', 0), (503, b'{"error":"kv_capacity_error"}', 1),
    (503, b'ordinary unavailable', 0), (502, b'bad gateway', 0)])
async def test_status_refusal_and_byte_identity(status, body, refusals):
    snapshots = []
    for enabled in (True, False):
        async with harness(status, body, enabled=enabled) as h:
            async with h.client.post(ROUTE, data=RAW) as response:
                snapshots.append((response.status, response.raw_headers, await response.read()))
            await asyncio.wait_for(h.completed.wait(), 2)
            assert h.seen == [RAW]
            if enabled:
                assert dict(h.metrics.http_requests) == {(ROUTE, str(status)): 1}
                assert sum(h.metrics.kv_refusals.values()) == refusals
                assert dict(h.metrics.duration_count) == {ROUTE: 1}
                assert dict(h.metrics.request_n) == {(ROUTE, 'n>1'): 1}
                assert SECRET not in h.metrics.render([])
                assert SECRET not in repr(vars(h.metrics))
            else:
                assert not h.metrics.http_requests
    assert snapshots[0] == snapshots[1]
    assert snapshots[0][0] == status and snapshots[0][2] == body


@pytest.mark.asyncio
async def test_stream_duration_only_after_eof_and_byte_identity():
    snapshots = []
    for enabled in (True, False):
        async with harness(stream=True, enabled=enabled) as h:
            async with h.client.post(ROUTE, data=RAW) as response:
                await asyncio.wait_for(h.started.wait(), 2)
                first = await response.content.readexactly(len(b'data: first\n\n'))
                assert not h.metrics.duration_count
                assert not h.metrics.duration_bucket
                assert not h.metrics.http_requests
                h.finish.set()
                rest = await response.read()
                await asyncio.wait_for(h.completed.wait(), 2)
                snapshots.append((response.status, response.raw_headers, first + rest))
                if enabled:
                    assert dict(h.metrics.duration_count) == {ROUTE: 1}
                    assert h.metrics.duration_bucket[(ROUTE, float('inf'))] == 1
                    assert dict(h.metrics.http_requests) == {(ROUTE, '200'): 1}
    assert snapshots[0] == snapshots[1]
    assert snapshots[0][2] == b'data: first\n\ndata: [DONE]\n\n'


@pytest.mark.asyncio
async def test_collector_failure_is_best_effort(monkeypatch):
    async with harness() as h:
        def broken(*args):
            raise RuntimeError('collector failure')
        monkeypatch.setattr(h.metrics, 'http_request', broken)
        monkeypatch.setattr(h.metrics, 'duration', broken)
        async with h.client.post(ROUTE, data=RAW) as response:
            assert response.status == 200
            assert await response.read() == b'\x00opaque\xff'
        await asyncio.wait_for(h.completed.wait(), 2)
        assert h.metrics.metrics_errors == 2
        assert 'pool_metrics_errors_total 2' in h.metrics.render([])


@pytest.mark.asyncio
async def test_non_completions_and_wrong_method_do_not_count():
    async with harness() as h:
        for path in ('/health', ROUTE):
            async with h.client.get(path) as response:
                assert response.status == 200
                assert await response.read() == b'health'
        assert not h.seen
        assert not h.metrics.http_requests
        assert not h.metrics.request_n
        assert not h.metrics.duration_count
        assert not h.metrics.kv_refusals
