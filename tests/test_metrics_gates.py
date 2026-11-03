import asyncio
from types import SimpleNamespace
import pytest
from aiohttp import web
from pool_proxy.app import App
from pool_proxy.metrics import Metrics
from conftest import make_cfg

ROUTE = '/v1/chat/completions'

@pytest.mark.asyncio
@pytest.mark.parametrize('gate', ['auth', 'startup', 'json', 'limit', 'error', 'cancel', 'retry'])
async def test_request_gates_and_denominator(gate, monkeypatch):
    p = App.__new__(App)
    p.metrics = Metrics()
    p.cfg = make_cfg([{'name':'fake','url':'http://127.0.0.1:1','np':1}], client_key_env='PP_TEST_KEY' if gate == 'auth' else None)
    p._started = gate != 'startup'
    monkeypatch.setenv('PP_TEST_KEY', 'secret')
    reads = []
    class Request:
        path = ROUTE
        method = 'POST'
        headers = {}
        async def read(self):
            reads.append(True)
            if gate == 'limit':
                raise web.HTTPRequestEntityTooLarge(max_size=1, actual_size=2)
            return b'not-json'
    async def handler(req):
        if gate == 'error':
            raise RuntimeError('test')
        if gate == 'cancel':
            raise asyncio.CancelledError()
        if gate == 'retry':
            # Multiple attempts must not create request-duration samples.
            assert not p.metrics.duration_count
            for _ in range(3):
                p.metrics.kv_refusal(ROUTE)
            return web.Response(status=503)
        return await p._client_auth(req, p.chat)
    expected = {'auth':401, 'startup':503, 'json':400, 'limit':413, 'error':500, 'cancel':500, 'retry':503}[gate]
    try:
        response = await asyncio.wait_for(p._request_metrics(Request(), handler), 1)
        assert response.status == expected
    except (web.HTTPRequestEntityTooLarge, RuntimeError, asyncio.CancelledError):
        assert gate in ('limit', 'error', 'cancel')
    assert sum(p.metrics.http_requests.values()) == sum(p.metrics.duration_count.values()) == 1
    assert p.metrics.http_requests[(ROUTE, str(expected))] == 1
    if gate in ('auth', 'startup'):
        assert not reads
        assert not p.metrics.request_n

@pytest.mark.asyncio
async def test_order_probe_no_body_io():
    p = App.__new__(App)
    p.metrics = Metrics()
    class Request:
        path = ROUTE
        method = 'POST'
        async def read(self):
            pytest.fail('observer read body before auth')
    async def auth(req):
        return web.Response(status=401)
    assert (await p._request_metrics(Request(), auth)).status == 401
