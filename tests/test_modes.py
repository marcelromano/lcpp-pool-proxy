import asyncio
import hashlib
import json
import time
import uuid
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web

from conftest import Fake, Proxy, make_cfg
from pool_proxy import config as cfgmod
from pool_proxy.app import App
from pool_proxy.modes import ModeManager
from pool_proxy.routerctl import ModelFailed, RouterCtl, RouterError, slot_filename

U = [{"role": "user", "content": "oi"}]
RUNNING = ("loaded", "loading", "sleeping")


class FakeRouter(Fake):
    """llama-server em router mode: /models, /models/load|unload, /slots (model+autoload), save/restore."""

    def __init__(self, name="router", models=("q27", "q35"), load_delay=0.05, unload_delay=0.02, **kw):
        super().__init__(name, **kw)
        self.st = {m: "unloaded" for m in models}
        self.failed: dict[str, bool] = {}
        self.fail_load: set[str] = set()
        self.no_slot_path: set[str] = set()      # modelos sem --slot-save-path: save/restore => 501
        self.load_delay, self.unload_delay = load_delay, unload_delay
        self.slots: dict[str, list[dict]] = {m: [] for m in models}
        self.calls: list[tuple] = []
        self.chat_hold: asyncio.Event | None = None
        self.chat_started = asyncio.Event()
        self.engine_epoch = "fake-router-" + uuid.uuid4().hex
        self.engine_epochs = {m: self.engine_epoch for m in models}
        self.engine_ids = {m: "router-" + m for m in models}
        self.last_acked = {}
        self.operations = {}
        self.slot_writer_gate: asyncio.Event | None = None
        self.slot_writer_started = asyncio.Event()
        self.slot_files_dir: Path | None = None
        self.slot_writer_task: asyncio.Task | None = None

    def extra_routes(self, app):
        app.router.add_get("/props", self._props)
        app.router.add_get("/models", self._models)
        app.router.add_post("/models/load", self._load)
        app.router.add_post("/models/unload", self._unload)
        app.router.add_get("/slots", self._slots)
        app.router.add_post("/slots/{id}", self._slot_action)
        app.router.add_get("/slot-operations", self._slot_capability)
        app.router.add_get("/slot-operations/{op_id}", self._slot_query)
        app.router.add_post("/slot-operations/{op_id}/ack", self._slot_ack)

    async def _slot_capability(self, r):
        model = r.query.get("model")
        self.calls.append(("op-capability", model, r.query.get("autoload")))
        if not self.engine_ids.get(model):
            return web.json_response({"error": "identity is not configured"}, status=501)
        return web.json_response({"capability": "op_id-terminal-v1",
                                  "engine_epoch": self.engine_epochs.get(model, self.engine_epoch),
                                  "engine_id": self.engine_ids.get(model, ""), "owner": "pool-proxy",
                                  "next_seq": self.last_acked.get(model, 0) + 1})

    async def _slot_query(self, r):
        self.calls.append(("op-query", r.query.get("model"), r.query.get("autoload")))
        op = self.operations.get((r.query.get("model"), r.match_info["op_id"]))
        if op:
            op = op["envelope"]
        return web.json_response(op or {"error": "unknown"}, status=200 if op else 404)

    async def _slot_ack(self, r):
        body = await r.json()
        model = body.get("model")
        if (body.get("engine_epoch") != self.engine_epochs.get(model, self.engine_epoch)
                or type(body.get("owner_seq")) is not int
                or body.get("owner") != "pool-proxy" or body.get("engine_id") != self.engine_ids.get(model)):
            return web.json_response({"error": "identity claims required"}, status=400)
        self.calls.append(("op-ack", body.get("model"), r.query.get("autoload")))
        op_id = r.match_info["op_id"]
        op = self.operations.get((model, op_id))
        env = op.get("envelope") if op else None
        if env and body["owner_seq"] != env["owner_seq"]:
            return web.json_response({"error": "sequence mismatch"}, status=409)
        if env and env["status"] not in {"completed", "failed", "cancelled_before_start"}:
            return web.json_response({"error": "not terminal"}, status=409)
        if op:
            self.last_acked[model] = max(self.last_acked.get(model, 0), env["owner_seq"])
            del self.operations[(model, op_id)]
        return web.Response(status=204)

    async def _props(self, r):
        return web.json_response({"role": "router", "max_instances": 2, "models_autoload": False})

    async def _models(self, r):
        data = []
        for m, s in self.st.items():
            status = {"value": s, "args": []}
            if self.failed.get(m):
                status.update(failed=True, exit_code=1)
            data.append({"id": m, "status": status})
        return web.json_response({"object": "list", "data": data})

    async def _load(self, r):
        m = (await r.json())["model"]
        self.calls.append(("load", m))
        if self.st[m] in RUNNING:
            return web.json_response({"error": "model is already running"}, status=400)
        self.st[m] = "loading"
        self.failed[m] = False
        asyncio.get_running_loop().create_task(self._finish_load(m))
        return web.json_response({"success": True})

    async def _finish_load(self, m):
        await asyncio.sleep(self.load_delay)
        if m in self.fail_load:
            self.st[m], self.failed[m] = "unloaded", True
        else:
            self.st[m] = "loaded"

    async def _unload(self, r):
        m = (await r.json())["model"]
        self.calls.append(("unload", m))
        if self.st[m] not in RUNNING:
            return web.json_response({"error": "model is not running"}, status=400)

        async def fin():
            await asyncio.sleep(self.unload_delay)
            self.st[m] = "unloaded"
        asyncio.get_running_loop().create_task(fin())
        return web.json_response({"success": True})

    async def _slots(self, r):
        m = r.query.get("model")
        self.calls.append(("slots", m, r.query.get("autoload")))
        if self.st.get(m) not in RUNNING:
            return web.json_response({"error": "model is not loaded"}, status=400)
        return web.json_response(self.slots[m])

    async def _slot_action(self, r):
        body = await r.json()
        if not body.get("model"):
            return web.json_response({"error": "model name is missing from the request"}, status=400)
        act, m = r.query["action"], body["model"]
        if act in ("save", "restore", "erase"):
            slot = int(r.match_info["id"])
            self.calls.append((act, m, slot, body.get("filename", ""), r.query.get("autoload")))
            if m in self.no_slot_path and act in ("save", "restore"):
                return web.json_response({"error": "slot save disabled"}, status=501)
            required = {"model", "op_id", "engine_epoch", "owner_seq", "owner", "engine_id"}
            if act in ("save", "restore"):
                required.add("filename")
            if (set(body) != required or body.get("owner") != "pool-proxy"
                    or body.get("engine_id") != self.engine_ids.get(m)):
                return web.json_response({"error": "wire shape or identity mismatch"}, status=409)
            op_id = body["op_id"]
            key = (m, op_id)
            if key in self.operations:
                if self.operations[key]["wire"] != body:
                    return web.json_response({"error": "operation replay mismatch"}, status=409)
                return web.json_response(self.operations[key]["envelope"])
            if body.get("engine_epoch") != self.engine_epoch or body.get("owner_seq") != self.last_acked.get(m, 0) + 1:
                return web.json_response({"error": "stale sequence or epoch"}, status=409)
            result = ({"n_saved": self.slots[m][slot].get("n_prompt_tokens", 0), "n_written": 128}
                      if act == "save" else {"n_restored": 128} if act == "restore" else {"n_erased": 1})
            env = {"op_id": op_id, "engine_id": self.engine_ids[m], "engine_epoch": self.engine_epoch,
                   "owner": "pool-proxy", "owner_seq": body["owner_seq"], "action": act,
                   "slot": slot, "filename": body.get("filename", ""),
                   "status": "running" if self.slot_writer_gate is not None and act == "save" else "completed",
                   "result": result}
            self.operations[key] = {"wire": dict(body), "envelope": env}
            if self.slot_writer_gate is not None and act == "save":
                self.slot_writer_started.set()
                self.slot_writer_task = asyncio.create_task(self._fake_engine_writer(env))
            return web.json_response(env)
        self.calls.append((act, m, int(r.match_info["id"]), body["filename"], r.query.get("autoload")))
        if self.st[m] != "loaded":
            return web.json_response({"error": "model is not loaded"}, status=400)
        return web.json_response({"id_slot": int(r.match_info["id"]), "filename": body["filename"]})

    async def _chat(self, request):
        body = await request.json()
        self.calls.append(("chat", body.get("model")))
        self.chat_started.set()
        if self.chat_hold is not None:
            await self.chat_hold.wait()
        if self.st.get(body.get("model")) != "loaded":
            return web.json_response({"error": "model is not loaded"}, status=400)
        return web.json_response({"backend": self.name, "model": body["model"], "choices": []})

    async def _fake_engine_writer(self, envelope):
        """Independent CPU fake writer: hold the temp file open until explicit release."""
        assert self.slot_files_dir is not None and self.slot_writer_gate is not None
        temp = self.slot_files_dir / (envelope["filename"] + ".tmp")
        final = self.slot_files_dir / envelope["filename"]
        with temp.open("wb") as writer:
            writer.write(b"writer-owned-temporary-snapshot")
            writer.flush()
            import os
            os.fsync(writer.fileno())
            await self.slot_writer_gate.wait()
        temp.replace(final)
        envelope["status"] = "completed"
        envelope["result"] = {"n_saved": 2048, "n_written": final.stat().st_size}

    def ops(self, *kinds):
        return [c[:2] for c in self.calls if c[0] in kinds]


POLICY = dict(switch_min_dwell_s=0, min_mode_s=0, eval_interval_s=0.05, poll_s=0.02, switch_timeout_s=5,
              drain_timeout_s=2, load_timeout_s=2, unload_timeout_s=2, fail_backoff_s=0.2,
              recover_interval_s=0.1, save_min_tokens=100)


def router_backend(fake, **kw):
    pol = {**POLICY, **kw.pop("mode_policy", {})}
    return {"name": fake.name, "url": fake.url, "np": 2, "router": True,
            "modes": {"m27": ["q27"], "m35": ["q35"]}, "default_mode": "m27",
            "model_engine_ids": {"q27": "router-q27", "q35": "router-q35"},
            "model_engine_owners": {"q27": "pool-proxy", "q35": "pool-proxy"},
            "mode_policy": pol, **kw}


async def until(pred, timeout=3.0):
    t0 = time.monotonic()
    while not pred():
        assert time.monotonic() - t0 < timeout, "timeout esperando condicao"
        await asyncio.sleep(0.01)


@pytest.fixture
async def rig():
    things = []

    async def make(router, others=(), **kw):
        await router.start()
        for o in others:
            await o.start()
        backends = [router_backend(router, **kw)]
        for o in others:
            backends.append({"name": o.name, "url": o.url, "overflow": True, "np": 2})
        p = await Proxy(App(make_cfg(backends, queue_timeout_s=5))).start()
        things.append((router, others, p))
        mgr = p.app.modes[router.name]
        await until(lambda: mgr.mode == "m27" and mgr.switching is None)
        return p, mgr

    yield make
    for router, others, p in things:
        await p.stop()
        for f in (router, *others):
            await f.stop()


def chat(p, model):
    return p.chat(U, body={"model": model})


async def test_initial_sync_loads_default_mode(rig):
    r = FakeRouter()
    p, mgr = await rig(r)
    assert r.st == {"q27": "loaded", "q35": "unloaded"}
    assert r.ops("load") == [("load", "q27")]
    st, backend, _, txt = await chat(p, "q27")
    assert (st, backend) == (200, "router") and '"q27"' in txt


async def test_full_switch_save_unload_load_restore(rig):
    r = FakeRouter()
    p, mgr = await rig(r)
    r.slots["q27"] = [{"id": 0, "n_prompt_tokens": 5000}, {"id": 1, "n_prompt_tokens": 10}]
    st, _, _, txt = await chat(p, "q35")
    assert st == 200 and '"q35"' in txt
    assert mgr.mode == "m35" and r.st == {"q27": "unloaded", "q35": "loaded"}
    fn = slot_filename("q27", 0)
    # ordem: save (so o slot com cache relevante, com model no corpo e autoload=false) -> unload -> load -> chat
    seq = [c[:4] if c[0] == "save" else c[:2] for c in r.calls
           if c[0] in ("save", "unload", "load", "chat")][1:]
    assert seq == [("save", "q27", 0, fn), ("unload", "q27"), ("load", "q35"), ("chat", "q35")]
    assert all(c[2] == "false" for c in r.calls if c[0] == "slots")
    assert all(c[4] == "false" for c in r.calls if c[0] in ("save", "restore"))
    assert not any(c[0] == "slots" and r.st[c[1]] != "loaded" and c[1] == "q35" for c in r.calls)
    # volta ao modo 27 => restore do arquivo salvo, no modelo certo
    mgr.pol.min_mode_s = 0
    st, _, _, _ = await chat(p, "q27")
    assert st == 200 and mgr.mode == "m27"
    assert ("restore", "q27", 0, fn, "false") in r.calls
    assert not any(c[0] == "save" and c[1] == "q35" for c in r.calls)   # q35 nao tinha cache
    m = p.app.metrics.render(p.app.pool.states.values())
    assert 'pool_mode_switch_total{backend="router",from="m27",to="m35",result="ok"} 1' in m
    assert 'pool_backend_mode{backend="router",mode="m27"} 1' in m
    assert 'pool_mode_slot_ops_total{backend="router",op="save",result="ok"} 1' in m


async def test_hysteresis_short_burst_does_not_switch():
    now = [0.0]
    pol = cfgmod.ModePolicy(switch_min_dwell_s=30, min_mode_s=100, switch_min_requests=2,
                            busy_threshold_requests=2, demand_window_s=300)
    cfg = cfgmod.BackendCfg(name="r", url="http://x", router=True, modes={"a": ["A"], "b": ["B"]},
                            default_mode="a", mode_policy=pol)
    mgr = ModeManager(cfg, ctl=None, clock=lambda: now[0])
    mgr.mode, mgr.mode_since = "a", 0.0
    burst = []
    for t in (1, 2, 3, 4, 5):                 # rajada de 5 pedidos em 4 s
        now[0] = t
        burst.append(mgr.record("B"))
    for ent in burst:                         # a rajada saiu do portao (desistiu/estourou a fila)
        ent[2] = False
    now[0] = 200                              # min_mode ok, mas a rajada nao persistiu
    assert mgr.decide() is None
    now[0] = 50
    mgr._demand.clear()
    for t in (50, 51):                        # persistiu menos que o dwell
        now[0] = t
        mgr.record("B")
    now[0] = 90
    assert mgr.decide() is None               # min_mode_s (100 s) ainda nao passou
    now[0] = 101
    mgr.record("B")                           # span 51 s >= dwell, requests >= 2, min_mode ok
    assert mgr.decide() == "b"
    mgr.active = 1                            # modo atual ocupado e acima do limiar
    for _ in range(3):
        mgr.record("A")
    assert mgr.decide() is None
    mgr.active = 0                            # ocioso: pode
    assert mgr.decide() == "b"


async def test_hysteresis_in_proxy_burst_not_switching(rig):
    r = FakeRouter()
    p, mgr = await rig(r, mode_policy={"switch_min_dwell_s": 30, "switch_timeout_s": 0.3})
    res = await asyncio.gather(*(chat(p, "q35") for _ in range(3)))
    assert all(x[0] == 503 for x in res)      # sem backend alternativo: 503 apos switch_timeout
    assert mgr.mode == "m27" and r.ops("unload") == []


async def test_load_failure_rolls_back_to_previous_mode(rig):
    r = FakeRouter()
    p, mgr = await rig(r, mode_policy={"switch_timeout_s": 1.0, "fail_backoff_s": 30})
    r.fail_load.add("q35")
    r.slots["q27"] = [{"id": 0, "n_prompt_tokens": 5000}]
    st, *_ = await chat(p, "q35")
    assert st == 503
    await until(lambda: mgr.mode == "m27" and mgr.switching is None and r.st["q27"] == "loaded")
    assert r.ops("unload", "load") == [("load", "q27"), ("unload", "q27"), ("load", "q35"), ("load", "q27")]
    assert ("restore", "q27", 0, slot_filename("q27", 0), "false") in r.calls   # cache de volta
    assert p.app.pool.states["router"].healthy and not mgr.broken
    m = p.app.metrics.render(p.app.pool.states.values())
    assert 'result="failed"' in m and 'result="rolled_back"' in m
    assert 'pool_mode_switch_failures_total{backend="router",step="load"} 1' in m
    assert (await chat(p, "q27"))[0] == 200


async def test_rollback_failure_marks_unhealthy_then_recovers(rig):
    r = FakeRouter()
    p, mgr = await rig(r, mode_policy={"switch_timeout_s": 0.5})
    r.fail_load.update({"q35", "q27"})
    st, *_ = await chat(p, "q35")
    assert st == 503
    await until(lambda: mgr.broken)
    assert not p.app.pool.states["router"].healthy
    await p.app.check_all()                   # /health do router e 200, mas segue unhealthy
    assert not p.app.pool.states["router"].healthy
    r.fail_load.clear()
    await until(lambda: not mgr.broken, timeout=5)   # tarefa de recuperacao
    assert r.st["q27"] == "loaded" and p.app.pool.states["router"].healthy
    assert (await chat(p, "q27"))[0] == 200


async def test_requests_during_switch_wait_or_spill(rig):
    ov = Fake("ov")
    r = FakeRouter(load_delay=0.5)
    p, mgr = await rig(r, [ov], mode_policy={"switch_timeout_s": 0.2})
    t35 = asyncio.create_task(chat(p, "q35"))      # dispara a troca (load de 0,5 s) e espera 0,2 s -> spill
    await until(lambda: mgr.switching == "m35")
    st, backend, reason, _ = await chat(p, "q27")  # modelo do modo que esta saindo: espera e cai no overflow
    assert (st, backend) == (200, "ov")
    st35 = await t35
    assert st35[0] == 200 and st35[1] == "ov"       # tambem estourou switch_timeout e foi para o overflow
    await until(lambda: mgr.mode == "m35")
    # com a troca concluida e janela longa o bastante, o pedido espera e sai no router
    p.app.modes["router"].pol.switch_timeout_s = 5
    assert (await chat(p, "q35"))[1] == "router"


async def test_queued_request_served_after_switch(rig):
    r = FakeRouter(load_delay=0.3)
    p, mgr = await rig(r)
    res = await asyncio.gather(chat(p, "q35"), chat(p, "q35"), chat(p, "q35"))
    assert [x[:2] for x in res] == [(200, "router")] * 3
    assert r.ops("load") == [("load", "q27"), ("load", "q35")]      # uma unica troca


async def test_drain_waits_for_inflight_before_save_and_unload(rig):
    r = FakeRouter()
    r.chat_hold = asyncio.Event()
    p, mgr = await rig(r)
    t27 = asyncio.create_task(chat(p, "q27"))
    await asyncio.wait_for(r.chat_started.wait(), 2)
    t35 = asyncio.create_task(chat(p, "q35"))
    await until(lambda: mgr.switching == "m35")
    await asyncio.sleep(0.3)
    assert r.ops("unload") == [] and r.st["q27"] == "loaded"    # em voo: nada e descarregado
    r.chat_hold.set()
    assert (await t27)[0] == 200
    assert (await t35)[0] == 200
    assert r.ops("unload") == [("unload", "q27")] and mgr.mode == "m35"


async def test_drain_timeout_aborts_switch_without_touching_models(rig):
    r = FakeRouter()
    r.chat_hold = asyncio.Event()
    p, mgr = await rig(r, mode_policy={"drain_timeout_s": 0.3, "switch_timeout_s": 1.0, "fail_backoff_s": 30})
    t27 = asyncio.create_task(chat(p, "q27"))
    await asyncio.wait_for(r.chat_started.wait(), 2)
    st, *_ = await chat(p, "q35")
    assert st == 503
    assert mgr.mode == "m27" and mgr.switching is None and r.ops("unload") == []
    assert 'step="drain"' in p.app.metrics.render(p.app.pool.states.values())
    r.chat_hold.set()
    assert (await t27)[0] == 200


async def test_model_alias_and_default_model(rig):
    r = FakeRouter()
    p, mgr = await rig(r, model_aliases={"qwen-grande": "q27"}, default_model="q27")
    for name in ("qwen-grande", "pool", ""):
        st, _, _, txt = await chat(p, name)
        assert st == 200 and '"q27"' in txt, name
    assert ("chat", "q27") in r.calls and not any(c[0] == "chat" and c[1] != "q27" for c in r.calls)


async def test_backend_without_router_is_unaffected_and_router_skipped_when_model_unknown(rig):
    other = Fake("plain")
    r = FakeRouter()
    p, mgr = await rig(r, [other])
    other_cfg = p.app.pool.states["plain"].cfg
    assert not other_cfg.router and p.app.pool.states["plain"].modes is None
    st, backend, _, _ = await chat(p, "modelo-que-o-router-nao-conhece")
    assert (st, backend) == (200, "plain")      # router nao serve o modelo e nao tem default_model


# ---- routerctl -----------------------------------------------------------------------------

async def test_routerctl_wait_loaded_raises_on_failed_load():
    r = await FakeRouter().start()
    async with aiohttp.ClientSession() as s:
        ctl = RouterCtl(lambda: s, r.url, poll_s=0.01)
        assert (await ctl.props())["role"] == "router"
        r.fail_load.add("q35")
        await ctl.load("q35")
        with pytest.raises(ModelFailed):
            await ctl.wait_loaded("q35", 2)
        with pytest.raises(RouterError) as e:
            await ctl.slots("q35")                   # descarregado: 400, nunca autocarrega
        assert e.value.status == 400
        await ctl.load("q27")
        await ctl.wait_loaded("q27", 2)
        with pytest.raises(RouterError):
            await ctl.load("q27")                    # ja roda: 400
        await ctl.unload("q27")
        assert (await ctl.wait_unloaded("q27", 2)).status == "unloaded"
    await r.stop()


def test_config_router_validation():
    bad = {"name": "r", "url": "http://x", "router": True}
    with pytest.raises(ValueError):
        cfgmod.from_dict({"backends": [bad]})
    ok = {**bad, "modes": {"a": ["A"], "b": ["B"]},
          "model_engine_ids": {"A": "engine-A", "B": "engine-B"},
          "model_engine_owners": {"A": "pool-proxy", "B": "pool-proxy"}}
    b = cfgmod.from_dict({"backends": [ok]}).backends[0]
    assert b.default_mode == "a" and b.mode_policy.min_mode_s == 600
    with pytest.raises(ValueError):
        cfgmod.from_dict({"backends": [{**ok, "default_model": "Z"}]})


async def test_np1_concurrent_foreign_requests_reach_min_requests_and_switch(rig):
    # regressao (teste real na 3090): com np=1 o portao de modo ficava depois do acquire; o 1o pedido
    # esperava o modelo segurando o unico slot e os demais nunca registravam demanda -> nenhuma troca
    r = FakeRouter(load_delay=0.2)
    p, mgr = await rig(r, np=1, mode_policy={"switch_min_requests": 3})
    res = await asyncio.gather(chat(p, "q35"), chat(p, "q35"), chat(p, "q35"))
    assert [x[:2] for x in res] == [(200, "router")] * 3
    assert mgr.mode == "m35"


def _unit_mgr(now, **pol_kw):
    pol = cfgmod.ModePolicy(**{**dict(switch_min_dwell_s=0, min_mode_s=0, switch_min_requests=3,
                                      busy_threshold_requests=4, demand_window_s=300), **pol_kw})
    cfg = cfgmod.BackendCfg(name="r", url="http://x", router=True,
                            modes={"a": ["A"], "b": ["B"], "c": ["C"]}, default_mode="a", mode_policy=pol)
    mgr = ModeManager(cfg, ctl=None, clock=lambda: now[0])
    mgr.mode, mgr.mode_since = "a", 0.0
    return mgr


def test_served_demand_does_not_count_and_tie_goes_to_current_mode():
    # oscilacao ao vivo: 3 pedidos B ja atendidos + 3 C esperando empatavam e vencia o 1o da lista
    now = [100.0]
    mgr = _unit_mgr(now)
    mgr.mode = "c"
    for _ in range(3):
        e = mgr.record("C")
        e[2] = False                          # C atendido ha poucos segundos
    for _ in range(3):
        mgr.record("B")                       # B esperando no portao
    assert mgr.decide() == "b"                # so o que espera conta; "a" (1o da lista) nao ganha empate
    for e in mgr._demand:
        e[2] = False                          # B tambem atendido
    assert mgr.decide() is None               # demanda atendida nao dispara troca


def test_tie_between_waiting_candidates_prefers_oldest_waiter():
    now = [100.0]
    mgr = _unit_mgr(now)
    mgr.mode = "c"
    for t, m in ((101, "B"), (102, "A"), (103, "B"), (104, "A"), (105, "B"), (106, "A")):
        now[0] = t
        mgr.record(m)
    now[0] = 110
    assert mgr.decide() == "b"                # 3 x 3: o mais antigo esperando (B), nao a ordem do dict
    assert mgr.modes and list(mgr.modes)[0] == "a"


def test_current_mode_in_use_wins_tie_with_margin():
    now = [100.0]
    mgr = _unit_mgr(now, switch_margin_requests=1)
    for _ in range(3):
        mgr.record("A")                       # modo atual servindo
    for _ in range(3):
        mgr.record("B")
    mgr.active = 1
    assert mgr.decide() is None               # empate com o modo atual em uso => nao troca
    mgr.record("B")
    assert mgr.decide() == "b"                # supera por 1
    mgr.pol.switch_margin_requests = 3
    assert mgr.decide() is None               # margem maior exige mais
    mgr.active = 0
    assert mgr.decide() == "b"                # ocioso: nao ha o que proteger


async def test_no_oscillation_after_switch_with_stale_demand(rig):
    # reproduz mineru1 -> m27 -> m35: demanda de q27 atendida nao pode empatar com a de q35
    r = FakeRouter()
    p, mgr = await rig(r, mode_policy={"switch_min_requests": 3})
    mgr.pol.min_mode_s = 0
    res = await asyncio.gather(*(chat(p, "q35") for _ in range(3)))
    assert [x[0] for x in res] == [200] * 3 and mgr.mode == "m35"
    res = await asyncio.gather(*(chat(p, "q27") for _ in range(3)))
    assert [x[0] for x in res] == [200] * 3 and mgr.mode == "m27"
    n = len(r.ops("load"))
    for _ in range(3):                        # q27 atendido: ocioso por varios ciclos de avaliacao
        st, *_ = await chat(p, "q27")
        assert st == 200
    await asyncio.sleep(0.5)
    assert mgr.mode == "m27" and len(r.ops("load")) == n     # sem troca espuria


async def test_501_slot_save_is_no_state_not_error(rig):
    r = FakeRouter()
    r.no_slot_path = {"q27"}
    p, mgr = await rig(r)
    r.slots["q27"] = [{"id": i, "n_prompt_tokens": 5000} for i in range(8)]
    assert (await chat(p, "q35"))[0] == 200 and mgr.mode == "m35"
    assert len([c for c in r.calls if c[0] == "save"]) == 1     # aprendeu no 1o 501, nao insiste nos 8 slots
    m = p.app.metrics.render(p.app.pool.states.values())
    assert 'op="save",result="error"' not in m
    assert 'op="save",result="unsupported"} 1' in m
    mgr.pol.min_mode_s = 0
    assert (await chat(p, "q27"))[0] == 200 and mgr.mode == "m27"
    assert not any(c[0] == "restore" for c in r.calls)
    assert 'result="error"' not in p.app.metrics.render(p.app.pool.states.values())
    n = len(r.calls)
    mgr.pol.min_mode_s = 0
    assert (await chat(p, "q35"))[0] == 200
    assert len([c for c in r.calls[n:] if c[0] == "save"]) == 0  # q27 lembrado sem estado


async def test_no_slot_save_config_skips_save_and_slots_call(rig):
    r = FakeRouter()
    p, mgr = await rig(r, no_slot_save=["q27"])
    r.slots["q27"] = [{"id": 0, "n_prompt_tokens": 5000}]
    assert (await chat(p, "q35"))[0] == 200 and mgr.mode == "m35"
    assert not any(c[0] in ("save", "restore") for c in r.calls)
    assert 'slot_ops_total' not in p.app.metrics.render(p.app.pool.states.values()).replace("# TYPE pool_mode_slot_ops_total counter", "")


def test_config_no_slot_save_validation():
    base = {"name": "r", "url": "http://x", "router": True, "modes": {"a": ["A"], "b": ["B"]},
            "model_engine_ids": {"A": "engine-A", "B": "engine-B"},
            "model_engine_owners": {"A": "pool-proxy", "B": "pool-proxy"}}
    b = cfgmod.from_dict({"backends": [{**base, "no_slot_save": ["A"]}]}).backends[0]
    assert b.no_slot_save == ["A"]
    with pytest.raises(ValueError):
        cfgmod.from_dict({"backends": [{**base, "no_slot_save": ["Z"]}]})


async def test_single_waiting_request_switches_after_dwell():
    now = [0.0]
    pol = cfgmod.ModePolicy(switch_min_dwell_s=10, min_mode_s=120, switch_min_requests=1,
                            busy_threshold_requests=4, demand_window_s=300)
    cfg = cfgmod.BackendCfg(name="r", url="http://x", router=True, modes={"a": ["A"], "b": ["B"]},
                            default_mode="a", mode_policy=pol)
    mgr = ModeManager(cfg, ctl=None, clock=lambda: now[0])
    mgr.mode, mgr.mode_since = "a", 0.0
    now[0] = 200
    mgr.record("B")                           # um pedido sozinho esperando no portao
    now[0] = 205
    assert mgr.decide() is None               # ainda dentro do dwell
    now[0] = 211
    assert mgr.decide() == "b"                # antes do fix (ts[-1]-ts[0]) nunca trocava


async def test_low_lease_never_records_demand_nor_enters_foreign_or_pending_switch():
    now = [100.0]
    mgr = _unit_mgr(now, switch_min_requests=1)
    assert mgr.try_enter_low("B") is None          # modelo fora do modo atual: nao entra
    assert not mgr._demand and mgr.decide() is None  # e nao registrou demanda (nunca provoca troca)
    lease = mgr.try_enter_low("A")
    assert lease is not None and mgr.active == 1
    lease.release()
    mgr.record("B")                                # producao esperando troca para B
    assert mgr.try_enter_low("A") is None          # low nao atrasa a troca pendente


async def test_low_request_for_parked_model_fails_fast(rig):
    p, _mgr = await rig(FakeRouter())                            # router no modo m27 (q27)
    s2, _, _, txt = await p.chat(U, body={"model": "q35"}, headers={"x-pool-priority": "low"})
    assert s2 == 409 and "modo atual" in txt


async def test_demanda_que_ainda_espera_sobrevive_a_purga_da_troca(rig):
    """a purga pos-troca ('demanda velha dos modelos que sairam') nao pode descartar
    quem AINDA esta esperando no portao. Sem a demanda, decide() nunca pedia a volta e o
    pedido expirava em switch_timeout_s devolvendo 503 na borda (27B com handoff seguido de
    35B imediato reproduzia isso com switch_timeout_s 120 e tambem com 180)."""
    r = FakeRouter()
    p, mgr = await rig(r)                 # comeca em m27 (q27)
    mgr.pol.min_mode_s = 0
    # cenario real: modo m27; chega um pedido do 27B (modelo que vai SAIR) e a troca para
    # m35 acontece no mesmo instante (no vivo, handoff 3090->MI50 pedindo m27 enquanto um 35B
    # esperava; a purga entao descartava o 35B que estava no portao)
    viva = mgr.record("q27")              # entrou no portao e continua esperando
    velha = mgr.record("q27"); velha[2] = False   # desistiu/expirou: nao conta mais
    assert viva[2] is True
    assert await mgr.switch("m35", "test") is True
    assert viva in mgr._demand            # quem espera continua registrado -> decide() pode voltar
    assert velha not in mgr._demand
async def test_low_switch_disabled_by_default_never_switches():
    now = [500.0]
    mgr = _unit_mgr(now, switch_min_requests=1)
    mgr.record_low("B")
    now[0] = 10_000
    assert mgr.decide() is None


async def test_low_switch_waits_dwell_and_needs_router_free_of_foreground():
    now = [500.0]
    mgr = _unit_mgr(now, switch_min_requests=1, low_switch=True, low_switch_dwell_s=60)
    ent = mgr.record_low("B")
    now[0] = 530
    assert mgr.decide() is None                    # dentro do dwell low
    now[0] = 561
    assert mgr.decide() == "b"
    mgr.record("A")                                # primeiro plano usou o modo atual na janela
    assert mgr.decide() is None
    mgr._demand.clear()
    mgr.active = 1                                 # algo em voo (inclusive low): nao troca
    assert mgr.decide() is None
    mgr.active = 0
    ent[2] = False                                 # o pedido low desistiu/foi atendido
    assert mgr.decide() is None


async def test_foreground_demand_wins_over_low_demand():
    now = [500.0]
    mgr = _unit_mgr(now, switch_min_requests=1, low_switch=True, low_switch_dwell_s=0)
    for _ in range(3):
        mgr.record_low("B")
    mgr.record("C")                                # primeiro plano esperando outro modo
    assert mgr.decide() == "c"


async def test_low_request_switches_router_when_enabled(rig):
    r = FakeRouter()
    p, mgr = await rig(r, mode_policy={"low_switch": True, "low_switch_dwell_s": 0})
    st, backend, reason, txt = await asyncio.wait_for(
        p.chat(U, body={"model": "q35"}, headers={"x-pool-priority": "low"}), 10)
    assert (st, backend, reason) == (200, "router", "low") and '"q35"' in txt
    assert mgr.mode == "m35" and r.st == {"q27": "unloaded", "q35": "loaded"}
    m = p.app.metrics.render(p.app.pool.states.values())
    assert 'pool_mode_switch_total{backend="router",from="m27",to="m35",result="ok"} 1' in m
    assert not mgr._low_demand or not any(w for _t, _m, w in mgr._low_demand)


async def test_crashed_model_marks_unhealthy_and_recovers(rig):
    r = FakeRouter()
    p, mgr = await rig(r, mode_policy={"alive_check_s": 0.05})
    assert (await chat(p, "q27"))[0] == 200
    r.st["q27"] = "unloaded"                       # filho do modelo morreu (ex. GGML_ASSERT)
    await until(lambda: mgr.broken or r.ops("load").count(("load", "q27")) >= 2)
    await until(lambda: not mgr.broken and r.st["q27"] == "loaded" and mgr.mode == "m27")
    assert r.ops("load").count(("load", "q27")) == 2   # recarregado pela recuperacao
    st, backend, _, _ = await chat(p, "q27")
    assert (st, backend) == (200, "router")
    m = p.app.metrics.render(p.app.pool.states.values())
    assert 'result="unhealthy"' in m
