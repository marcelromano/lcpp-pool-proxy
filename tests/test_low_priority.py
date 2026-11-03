import asyncio

from conftest import Fake

U = [{"role": "user", "content": "oi"}]
LOW = {"headers": {"x-pool-priority": "low"}}


async def test_low_uses_idle_backend(env):
    a = Fake("a")
    p = await env([a], [{}], low_idle_s=0, low_poll_s=0.05)
    st, backend, reason, _ = await p.chat(U, **LOW)
    assert (st, backend, reason) == (200, "a", "low")


async def test_low_waits_while_backend_busy_then_runs(env):
    a = Fake("a")
    p = await env([a], [{}], low_idle_s=0, low_poll_s=0.05)
    sa = p.app.pool.states["a"]
    sa.inflight = 1                      # pedido de producao em curso
    task = asyncio.create_task(p.chat(U, **LOW))
    await asyncio.sleep(0.3)
    assert not task.done() and not a.requests
    sa.inflight = 0
    st, backend, _, _ = await asyncio.wait_for(task, 5)
    assert (st, backend) == (200, "a")


async def test_low_respects_idle_gap(env):
    a = Fake("a")
    p = await env([a], [{}], low_idle_s=0.5, low_poll_s=0.05)
    p.app.pool.states["a"].idle_since = asyncio.get_running_loop().time() + 10**9  # acabou de atender
    import time
    p.app.pool.states["a"].idle_since = time.monotonic()
    task = asyncio.create_task(p.chat(U, **LOW))
    await asyncio.sleep(0.2)
    assert not task.done()
    assert (await asyncio.wait_for(task, 5))[0] == 200


async def test_low_times_out_with_503(env):
    a = Fake("a")
    p = await env([a], [{}], low_idle_s=0, low_poll_s=0.05, low_wait_s=0.3)
    p.app.pool.states["a"].healthy = False   # janela de GPU: backend fora
    st, _, _, _ = await p.chat(U, **LOW)
    assert st == 503 and not a.requests


async def test_low_after_low_skips_idle_gap(env):
    a = Fake("a")
    p = await env([a], [{}], low_idle_s=5, low_poll_s=0.05)
    p.app.pool.states["a"].idle_since = -10**9           # primeiro plano ocioso ha muito tempo
    for _ in range(3):                                   # turnos seguidos de um worker: sem esperar 5 s cada
        st, backend, reason, _ = await asyncio.wait_for(p.chat(U, **LOW), 2)
        assert (st, backend, reason) == (200, "a", "low")


async def test_low_max_inflight_serializes_low(env):
    a = Fake("a", delay=0.3)
    p = await env([a], [{"np": 2}], low_idle_s=0, low_poll_s=0.02, low_max_inflight=1)
    r = await asyncio.gather(p.chat(U, **LOW), p.chat(U, **LOW))
    assert [x[0] for x in r] == [200, 200] and a.max_active == 1
    assert p.app.pool.states["a"].low_inflight == 0


async def test_low_two_slots_when_allowed(env):
    a = Fake("a", delay=0.3)
    p = await env([a], [{"np": 2}], low_idle_s=0, low_poll_s=0.02, low_max_inflight=2)
    await asyncio.gather(p.chat(U, **LOW), p.chat(U, **LOW))
    assert a.max_active == 2


async def test_foreground_still_opens_gap_after_low_chain(env):
    a = Fake("a")
    p = await env([a], [{}], low_idle_s=0.5, low_poll_s=0.05)
    p.app.pool.states["a"].idle_since = -10**9
    assert (await p.chat(U, **LOW))[0] == 200
    assert (await p.chat(U))[0] == 200                   # primeiro plano: reinicia o vao
    task = asyncio.create_task(p.chat(U, **LOW))
    await asyncio.sleep(0.2)
    assert not task.done()
    assert (await asyncio.wait_for(task, 5))[0] == 200


async def test_normal_request_unaffected(env):
    a = Fake("a")
    p = await env([a], [{}])
    st, backend, reason, _ = await p.chat(U)
    assert (st, backend) == (200, "a") and reason != "low"


class Loading(Fake):
    """Engine recarregando: responde 503 "Loading model" nas primeiras `n` chamadas."""

    def __init__(self, name, n=2, **kw):
        super().__init__(name, **kw)
        self.n = n

    async def _chat(self, request):
        if self.n > 0:
            self.n -= 1
            await request.read()
            from aiohttp import web
            return web.json_response({"error": {"message": "Loading model"}}, status=503)
        return await super()._chat(request)


async def test_low_retries_upstream_503_until_served(env):
    a = Loading("a", n=2)
    p = await env([a], [{}], low_idle_s=0, low_poll_s=0.02)
    st, backend, reason, _ = await asyncio.wait_for(p.chat(U, **LOW), 5)
    assert (st, backend, reason) == (200, "a", "low") and a.n == 0
    m = p.app.metrics.render(p.app.pool.states.values())
    assert 'status="low_retry"} 2' in m


async def test_low_gives_up_503_after_deadline(env):
    a = Loading("a", n=10**6)
    p = await env([a], [{}], low_idle_s=0, low_poll_s=0.02, low_wait_s=0.3)
    st, _, _, _ = await asyncio.wait_for(p.chat(U, **LOW), 5)
    assert st == 503


async def test_foreground_503_still_passes_through(env):
    a = Loading("a", n=1)
    p = await env([a], [{}])
    assert (await p.chat(U))[0] == 503       # so a faixa low repete; primeiro plano ve o erro como antes


async def test_low_prefers_backend_with_prefix_and_records_affinity(env):
    a, b = Fake("a"), Fake("b")
    p = await env([a, b], [{}, {}], low_idle_s=0, low_poll_s=0.02)
    msgs = [{"role": "user", "content": "x" * 40000}, {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "segue"}]
    st, first, _, _ = await p.chat(msgs, **LOW)
    assert st == 200
    msgs2 = msgs + [{"role": "assistant", "content": "ok2"}, {"role": "user", "content": "mais"}]
    for _ in range(3):                                  # turnos seguintes ficam no dono do prefixo
        st, backend, _, _ = await p.chat(msgs2, **LOW)
        assert (st, backend) == (200, first)
