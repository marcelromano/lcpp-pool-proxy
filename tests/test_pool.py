import asyncio
import json

from conftest import Fake

U = [{"role": "user", "content": "oi"}]


async def test_cost_prefers_faster_decode(env):
    a, b = Fake("lento"), Fake("rapido")
    p = await env([a, b], [{"decode_tps": 30}, {"decode_tps": 68}])
    st, backend, reason, _ = await p.chat(U, body={"max_tokens": 200})
    assert (st, backend, reason) == (200, "rapido", "cost")


async def test_weight_biases_cost(env):
    a, b = Fake("a"), Fake("b")
    p = await env([a, b], [{"weight": 1}, {"weight": 3}])
    assert (await p.chat(U))[1] == "b"


async def test_affinity_keeps_session_on_same_backend(env):
    a, b = Fake("a"), Fake("b")
    p = await env([a, b], [{"decode_tps": 30}, {"decode_tps": 68}])
    conv = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q1"}]
    _, first, _, _ = await p.chat(conv)
    assert first == "b"
    # b fica mais caro por fila fabricada; a afinidade ainda vence enquanto couber na espera maxima
    p.app.pool.states["b"].avg_s = 5.0
    conv2 = conv + [{"role": "assistant", "content": "r1"}, {"role": "user", "content": "q2"}]
    _, second, reason, _ = await p.chat(conv2)
    assert (second, reason) == ("b", "affinity")
    # conversa diferente nao herda
    _, other, reason, _ = await p.chat([{"role": "user", "content": "outra"}])
    assert reason == "cost"


async def test_affinity_dropped_when_wait_too_long(env):
    a, b = Fake("a"), Fake("b")
    p = await env([a, b], [{"decode_tps": 30}, {"decode_tps": 68}], affinity_max_wait_s=1)
    conv = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "r"},
            {"role": "user", "content": "q2"}]
    await p.chat(conv)
    sb = p.app.pool.states["b"]
    sb.inflight, sb.avg_s = 1, 100.0   # espera estimada = 100 s
    _, backend, reason, _ = await p.chat(conv + [{"role": "assistant", "content": "x"},
                                                 {"role": "user", "content": "q3"}])
    sb.inflight = 0
    assert backend == "a" and reason != "affinity"


async def test_direct_backend_models_filter(env):
    a, b = Fake("lento"), Fake("rapido")
    p = await env([a, b], [{"decode_tps": 30}, {"decode_tps": 68, "models": ["outro"]}])
    st, backend, _, _ = await p.chat(U)
    assert (st, backend) == (200, "lento")
    assert not b.requests


async def test_image_goes_only_to_vision_backend(env):
    a, b = Fake("texto"), Fake("visao")
    p = await env([a, b], [{"decode_tps": 100}, {"decode_tps": 10, "vision": True}])
    msg = [{"role": "user", "content": [{"type": "text", "text": "veja"},
                                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]
    st, backend, reason, _ = await p.chat(msg)
    assert (st, backend, reason) == (200, "visao", "vision")
    assert not a.requests


async def test_image_without_vision_backend_is_503(env):
    p = await env([Fake("texto")], [{}])
    msg = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]
    assert (await p.chat(msg))[0] == 503


async def test_long_cold_prompt_goes_to_highest_prefill(env):
    a, b = Fake("mi50"), Fake("rtx")
    p = await env([a, b], [{"prefill_tps": 140, "decode_tps": 200}, {"prefill_tps": 1000, "decode_tps": 30}],
                  long_prompt_tokens=1000)
    st, backend, reason, _ = await p.chat([{"role": "user", "content": "x" * 10000}], body={"max_tokens": 50})
    assert (backend, reason) == ("rtx", "long_cold")
    # curto: custo domina e escolhe o de decode rapido
    assert (await p.chat(U, body={"max_tokens": 500}))[1] == "mi50"


async def test_context_limit_excludes_backend(env):
    a, b = Fake("curto"), Fake("longo")
    p = await env([a, b], [{"ctx_max": 100, "decode_tps": 500}, {"ctx_max": 0}], long_prompt_tokens=10**9)
    assert (await p.chat([{"role": "user", "content": "y" * 3500}]))[1] == "longo"


async def test_concurrency_limit_is_np(env):
    a = Fake("a", delay=0.2)
    p = await env([a], [{"np": 2}])
    res = await asyncio.gather(*(p.chat([{"role": "user", "content": f"m{i}"}]) for i in range(6)))
    assert all(r[0] == 200 for r in res)
    assert a.max_active == 2


async def test_spill_to_overflow_on_estimated_wait(env):
    a, o = Fake("main", delay=0.3), Fake("nuvem")
    p = await env([a, o], [{}, {"overflow": True}], spill_after_s=5, default_service_s=10)
    res = await asyncio.gather(*(p.chat([{"role": "user", "content": f"m{i}"}]) for i in range(4)))
    backends = sorted(r[1] for r in res)
    assert backends.count("main") == 1 and backends.count("nuvem") == 3
    assert all(r[0] == 200 for r in res)
    assert a.max_active == 1


async def test_spill_on_queue_timeout(env):
    a, o = Fake("main", delay=1.0), Fake("nuvem")
    p = await env([a, o], [{}, {"overflow": True}], spill_after_s=1000, queue_timeout_s=0.2)
    r = await asyncio.gather(p.chat([{"role": "user", "content": "1"}]), p.chat([{"role": "user", "content": "2"}]))
    got = {x[1]: x[2] for x in r}
    assert got == {"main": "cost", "nuvem": "spill_timeout"} or set(got) == {"main", "nuvem"}
    assert all(x[0] == 200 for x in r)


async def test_queue_timeout_without_overflow_is_503(env):
    a = Fake("main", delay=1.0)
    p = await env([a], [{}], queue_timeout_s=0.2)
    r = await asyncio.gather(p.chat([{"role": "user", "content": "1"}]), p.chat([{"role": "user", "content": "2"}]))
    assert sorted(x[0] for x in r) == [200, 503]


async def test_backend_failure_retries_on_other_and_marks_down(env):
    a, b = Fake("morto"), Fake("vivo")
    p = await env([a, b], [{"decode_tps": 100}, {"decode_tps": 10}])
    await a.stop()
    a.runner = None
    st, backend, _, _ = await p.chat(U)
    assert (st, backend) == (200, "vivo")
    assert p.app.pool.states["morto"].healthy is False
    # proximas requisicoes nem tentam o morto
    assert (await p.chat(U))[1] == "vivo"


async def test_all_backends_down_is_503(env):
    a = Fake("a")
    p = await env([a], [{}])
    await a.stop()
    a.runner = None
    assert (await p.chat(U))[0] == 503
    async with p.http.get(p.url + "/health") as r:
        assert r.status == 503


async def test_health_check_removes_and_restores(env):
    a, b = Fake("a"), Fake("b")
    p = await env([a, b], [{}, {}], health_failures=2)
    await asyncio.sleep(0.2)   # deixa a checagem inicial do startup terminar
    a.health_status = 500
    await p.app.check_all()
    assert p.app.pool.states["a"].healthy is True   # 1 falha so
    await p.app.check_all()
    assert p.app.pool.states["a"].healthy is False
    assert (await p.chat(U))[1] == "b"
    a.health_status = 200
    await p.app.check_all()
    assert p.app.pool.states["a"].healthy is True


async def test_stream_passthrough_and_header_policy(env, monkeypatch):
    monkeypatch.setenv("KEY_A", "segredo-a")
    a = Fake("a", delay=0.1, chunks=3)
    p = await env([a], [{"key_env": "KEY_A"}])
    body = {"model": "pool", "messages": U, "stream": True}
    async with p.http.post(p.url + "/v1/chat/completions", json=body,
                           headers={"X-Conversation-Id": "c1", "Authorization": "Bearer do-cliente",
                                    "X-Outro": "ok"}) as r:
        assert r.status == 200
        assert r.headers["Content-Type"].startswith("text/event-stream")
        txt = await r.text()
    events = [l for l in txt.split("\n") if l.startswith("data: ")]
    assert len(events) == 4 and events[-1] == "data: [DONE]"
    h = {k.lower(): v for k, v in a.requests[0]["headers"].items()}
    assert "x-conversation-id" not in h
    assert h["authorization"] == "Bearer segredo-a"
    assert h["x-outro"] == "ok"
    assert a.requests[0]["body"] == body   # corpo repassado intacto


async def test_non_stream_body_forwarded(env):
    a = Fake("a")
    p = await env([a], [{}])
    st, _, _, txt = await p.chat(U, body={"temperature": 0.2, "max_tokens": 7})
    assert st == 200 and json.loads(txt)["backend"] == "a"
    assert a.requests[0]["body"]["temperature"] == 0.2


async def test_client_disconnect_closes_upstream(env):
    a = Fake("a", delay=30, chunks=100)
    p = await env([a], [{}])
    body = {"model": "pool", "messages": U, "stream": True}
    r = await p.http.post(p.url + "/v1/chat/completions", json=body)
    await r.content.readany()   # primeiro chunk chegou
    r.close()
    await asyncio.wait_for(a.disconnected.wait(), 5)
    for _ in range(50):
        if p.app.pool.states["a"].inflight == 0:
            break
        await asyncio.sleep(0.05)
    assert p.app.pool.states["a"].inflight == 0


async def test_client_disconnect_before_first_byte(env):
    a = Fake("a", delay=30)   # nao-stream: nenhuma escrita ate o fim
    p = await env([a], [{}])
    task = asyncio.create_task(p.chat(U))
    for _ in range(50):
        if a.active:
            break
        await asyncio.sleep(0.05)
    task.cancel()
    await p.http.close()
    p.http = __import__("aiohttp").ClientSession()
    await asyncio.wait_for(a.disconnected.wait(), 5)


async def test_models_health_metrics(env):
    a, b = Fake("a"), Fake("b")
    p = await env([a, b], [{}, {}])
    await p.chat(U)
    async with p.http.get(p.url + "/v1/models") as r:
        assert (await r.json())["data"][0]["id"] == "pool"
    async with p.http.get(p.url + "/health") as r:
        d = await r.json()
        assert r.status == 200 and sorted(d["backends_up"]) == ["a", "b"]
    async with p.http.get(p.url + "/metrics") as r:
        m = await r.text()
    assert 'pool_requests_total{backend="' in m
    assert "pool_route_decisions_total" in m and "pool_queue_depth" in m
    assert 'pool_backend_up{backend="a"} 1' in m
    assert "pool_request_seconds_count" in m


async def test_hooks_slot_probe_hint(env):
    from pool_proxy.hooks import NullHandoff

    class Probe:
        async def find_owner(self, backends, messages):
            return "a"

    a, b = Fake("a"), Fake("b")
    p = await env([a, b], [{"decode_tps": 10}, {"decode_tps": 100}])
    p.app.probe = Probe()
    p.app.handoff = NullHandoff()
    assert (await p.chat(U))[1:3] == ("a", "affinity")


async def test_bad_json_is_400(env):
    p = await env([Fake("a")], [{}])
    async with p.http.post(p.url + "/v1/chat/completions", data=b"{nao") as r:
        assert r.status == 400


async def test_client_auth_required_when_configured(monkeypatch):
    import aiohttp
    from conftest import Fake, Proxy, make_cfg
    from pool_proxy.app import App
    monkeypatch.setenv("POOL_CLIENT_KEY_T", "s3cret")
    f = await Fake("a").start()
    cfg = make_cfg([{"name": "a", "url": f.url}])
    cfg.settings.client_key_env = "POOL_CLIENT_KEY_T"
    p = await Proxy(App(cfg)).start()
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(p.url + "/health") as r:
                assert r.status in (200, 503)
            async with s.get(p.url + "/v1/models") as r:
                assert r.status == 401
            async with s.get(p.url + "/v1/models", headers={"Authorization": "Bearer errada"}) as r:
                assert r.status == 401
            async with s.get(p.url + "/v1/models", headers={"Authorization": "Bearer s3cret"}) as r:
                assert r.status == 200
    finally:
        await p.stop(); await f.stop()
