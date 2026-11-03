"""Handoff com destino em router mode: modelo carregado/por carregar, falha de carga, modelo errado."""
import asyncio
import hashlib
import json

import pytest
from aiohttp import web

from conftest import Proxy, be, make_cfg
from pool_proxy.app import App
from pool_proxy.config import from_dict
from test_handoff import SlotEngine
from test_modes import POLICY, FakeRouter, until

U = [{"role": "user", "content": "a" * 60}]
GGUF = "/models/27b.gguf"


class RouterTarget(FakeRouter):
    """Router falso cujo filho q27 tambem fala o protocolo de handoff (props/template/tokenize/slots)."""

    def __init__(self, directory, **kw):
        super().__init__("mi50", models=("q27", "q35"), **kw)
        self.directory = directory
        self.tokens = list(range(16))
        self.gguf = GGUF
        self.preset = {"cache-type-k": "q8_0", "cache-type-v": "q8_0", "kv-unified": "true"}
        self.events = []
        self.np = 1
        self.fail_restore = False

    def extra_routes(self, app):
        super().extra_routes(app)
        app.router.add_post("/apply-template", self._template)
        app.router.add_post("/tokenize", self._tokenize)

    def _check(self, model, autoload):
        if autoload != "false":
            return web.json_response({"error": "autoload must be false"}, status=500)
        if self.st.get(model) != "loaded":
            return web.json_response({"error": "model is not loaded"}, status=400)

    async def _props(self, r):
        model = r.query.get("model")
        if not model:
            return await super()._props(r)
        self.events.append(("props", model))
        return self._check(model, r.query.get("autoload")) or web.json_response({"model_path": self.gguf})

    async def _models(self, r):
        resp = await super()._models(r)
        import json
        data = json.loads(resp.text)
        for m in data["data"]:
            m["status"]["preset"] = f"[{m['id']}]\n" + "".join(f"{k} = {v}\n" for k, v in self.preset.items())
        return web.json_response(data)

    async def _slots(self, r):
        model = r.query.get("model")
        bad = self._check(model, r.query.get("autoload")) if model else None
        if bad is not None:
            return bad
        self.events.append(("slots", model))
        return web.json_response([{"id": i, "is_processing": False, "has_media": False, "n_cached_tokens": 0}
                                  for i in range(self.np)])

    async def _template(self, r):
        body = await r.json()
        self.events.append(("template", body.get("model")))
        return self._check(body.get("model"), r.query.get("autoload")) or web.json_response({"prompt": "templated"})

    async def _tokenize(self, r):
        body = await r.json()
        return self._check(body.get("model"), r.query.get("autoload")) or web.json_response({"tokens": self.tokens})

    async def _slot_action(self, r):
        body = await r.json()
        op, model = r.query["action"], body.get("model")
        bad = self._check(model, r.query.get("autoload"))
        if bad is not None:
            return bad
        op_id = body.get("op_id")
        required = {"model", "op_id", "engine_epoch", "owner_seq", "owner", "engine_id"}
        if op in ("save", "restore"):
            required.add("filename")
        if (set(body) != required or body.get("owner") != "pool-proxy"
                or body.get("engine_id") != self.engine_ids.get(model)
                or not op_id or body.get("engine_epoch") != self.engine_epoch
                or body.get("owner_seq") != self.last_acked.get(model, 0) + 1):
            return web.json_response({"error": "operation identity rejected"}, status=409)
        key = (model, op_id)
        if key in self.operations:
            previous = self.operations[key]
            if previous["wire"] != body:
                return web.json_response({"error": "operation retry mismatch"}, status=409)
            return web.json_response(previous["envelope"])
        self.events.append((op, model, body.get("filename", "")))
        if op == "restore":
            if self.fail_restore:
                result = {"op_id": op_id, "engine_id": self.engine_ids[model], "engine_epoch": self.engine_epoch,
                          "owner": "pool-proxy", "owner_seq": body["owner_seq"], "action": op,
                          "slot": int(r.match_info["id"]), "filename": body.get("filename", ""),
                          "status": "failed", "error": {"message": "x"}}
                self.operations[key] = {"wire": dict(body), "envelope": result}
                return web.json_response(result)
            assert (self.directory / body["filename"]).read_bytes() == b"state"
            payload = {"n_restored": len(self.tokens) - 1}
        else:
            return web.json_response({"error": "unexpected"}, status=409)
        result = {"op_id": op_id, "engine_id": self.engine_ids[model], "engine_epoch": self.engine_epoch,
                  "owner": "pool-proxy", "owner_seq": body["owner_seq"], "action": op,
                  "slot": int(r.match_info["id"]), "filename": body.get("filename", ""),
                  "status": "completed", "result": payload}
        self.operations[key] = {"wire": dict(body), "envelope": result}
        return web.json_response(result)


@pytest.fixture
async def rig(tmp_path):
    live = []

    async def make(initial="m27", handoff_models=("pool",), policy=None, settings=None, **backend):
        src = tmp_path / "source"
        dst = tmp_path / "kvslots-q27"
        src.mkdir(exist_ok=True)
        dst.mkdir(exist_ok=True)
        gguf = tmp_path / "27b.gguf"
        gguf.write_bytes(b"same model")
        a = await SlotEngine("3090", src).start()
        r = RouterTarget(dst)
        await r.start()
        pol = {**POLICY, **(policy or {})}
        router_cfg = {"name": "mi50", "url": r.url, "np": 1, "router": True, "prefill_tps": 100, "decode_tps": 30,
                      "model_engine_ids": {"q27": "router-q27", "q35": "router-q35"},
                      "model_engine_owners": {"q27": "pool-proxy", "q35": "pool-proxy"},
                      "slotops_ledger_dir": str(tmp_path / "router-ledger"),
                      "modes": {"m27": ["q27"], "m35": ["q35"]}, "default_mode": initial,
                      "default_model": "q27", "model_aliases": {"pool": "q27", "other": "q35"},
                      "mode_policy": pol, "handoff_model_path": GGUF, "handoff_gguf_path": str(gguf),
                      "handoff_state_format": "mainline-q8-q8", "handoff_model": "q27",
                      "handoff_preset": {"cache-type-k": "q8_0", "cache-type-v": "q8_0", "kv-unified": "true"}, **backend}
        backends = [be(a, prefill_tps=1000, handoff_model_path=a.gguf, handoff_gguf_path=str(gguf),
                       handoff_state_format="mainline-q8-q8"), router_cfg]
        cfg = make_cfg(backends, models=["pool", "other"], probe_enabled=False, long_prompt_tokens=1,
                       handoff_enabled=True, handoff_source="3090", handoff_target="mi50",
                       handoff_models=list(handoff_models), handoff_min_tokens=8, handoff_transport="copy",
                       handoff_source_dir=str(src), handoff_target_dir=str(dst), **(settings or {}))
        p = await Proxy(App(cfg)).start()
        live.append((p, a, r))
        mgr = p.app.modes["mi50"]
        await until(lambda: mgr.mode == initial and mgr.switching is None)
        r.calls.clear()
        for _ in range(400):
            if p.app.cold_handoff.ready_digest(str(gguf)) is not None:
                break
            await asyncio.sleep(0.005)
        else:
            raise AssertionError("hash")
        return p, a, r, mgr

    yield make
    for p, a, r in live:
        await p.stop()
        await a.stop()
        await r.stop()


def ops(r, *kinds):
    return [c[:2] for c in r.calls if c[0] in kinds]


async def test_router_target_model_loaded(rig):
    p, a, r, mgr = await rig("m27")
    status, backend, reason, _ = await p.chat(U)
    assert (status, backend, reason) == (200, "mi50", "handoff")
    assert not ops(r, "load", "unload")
    # tudo com o modelo canonico e autoload=false; restore no arquivo do diretorio do modelo
    assert ("props", "q27") in r.events and ("slots", "q27") in r.events
    restore = next(e for e in r.events if e[0] == "restore")
    assert restore[1] == "q27" and restore[2].startswith("pp-handoff-")
    assert ("chat", "q27") in r.calls
    assert mgr.active == 0
    assert not list(a.directory.iterdir()) and not list(r.directory.iterdir())
    assert p.app.metrics.handoffs["ok"] == 1


async def test_router_target_needs_load(rig):
    p, a, r, mgr = await rig("m35")
    assert r.st == {"q27": "unloaded", "q35": "loaded"}
    status, backend, reason, _ = await p.chat(U)
    assert (status, backend, reason) == (200, "mi50", "handoff")
    assert r.st == {"q27": "loaded", "q35": "unloaded"} and mgr.mode == "m27"
    # carga antes de qualquer restore e nunca dois modelos de GPU ao mesmo tempo
    assert ops(r, "load", "unload") == [("unload", "q35"), ("load", "q27")]
    assert any(e[0] == "restore" for e in r.events)   # o router falso recusa restore com o modelo descarregado
    assert mgr.active == 0


@pytest.mark.parametrize("overlap", [True, False])
async def test_router_target_load_failure_falls_back(rig, overlap):
    p, a, r, mgr = await rig("m35", settings={"handoff_switch_overlap": overlap})
    r.fail_load.add("q27")
    status, backend, reason, _ = await p.chat(U)
    assert status == 200 and backend == "3090" and reason != "handoff"
    assert p.app.metrics.handoffs["model_unavailable"] == 1
    assert not any(e[0] == "restore" for e in r.events)
    if overlap:
        # o prefill da origem terminou mesmo assim: o fallback encontra o prefixo quente na 3090
        assert any(e[0] == "prefill" for e in a.events)
    else:
        assert not a.events
    await until(lambda: r.st["q35"] == "loaded" and mgr.switching is None)   # rollback: modo original de pe
    assert mgr.active == 0 and all(not st.handoff_busy and st.inflight == 0 for st in p.app.pool.states.values())


async def test_overlap_load_runs_during_prefill(rig):
    """A carga do modelo comeca com o prefill da origem em curso, e o restore so depois dos dois."""
    p, a, r, mgr = await rig("m35")
    r.load_delay = 0.2
    a.pause = asyncio.Event()
    task = asyncio.create_task(p.chat(U))
    await asyncio.wait_for(a.prefilling.wait(), 5)
    await until(lambda: r.st["q27"] == "loaded")          # modelo pronto enquanto a origem ainda faz prefill
    assert not any(e[0] == "restore" for e in r.events)
    assert ("props", "q27") not in r.events                # verificacoes do destino esperam a lease
    a.pause.set()
    status, backend, reason, _ = await task
    assert (status, backend, reason) == (200, "mi50", "handoff") and mgr.active == 0
    m = p.app.metrics
    assert m.stage_count["lease_wait"] == 1 and m.stage_sum["lease_wait"] == 0.0
    assert m.stage_count["prefill_save"] == 1 and m.stage_count["restore"] == 1


async def test_overlap_slow_load_is_waited(rig):
    p, a, r, mgr = await rig("m35")
    r.load_delay = 0.6
    status, backend, reason, _ = await p.chat(U)
    assert (status, backend, reason) == (200, "mi50", "handoff")
    assert p.app.metrics.stage_sum["lease_wait"] > 0.3     # a troca durou mais que o prefill: so o excedente aparece


async def test_no_overlap_loads_before_prefill(rig):
    p, a, r, mgr = await rig("m35", settings={"handoff_switch_overlap": False})
    r.load_delay = 0.2
    a.pause = asyncio.Event()
    task = asyncio.create_task(p.chat(U))
    await asyncio.wait_for(a.prefilling.wait(), 5)
    assert r.st["q27"] == "loaded"                          # a lease (carga) precedeu o prefill
    a.pause.set()
    assert (await task)[2] == "handoff"
    assert "lease_wait" not in p.app.metrics.stage_count


async def test_overlap_cancel_releases_lease(rig):
    """Cliente some durante o prefill: lease e reserva do par voltam."""
    p, a, r, mgr = await rig("m35")
    a.pause = asyncio.Event()
    task = asyncio.create_task(p.chat(U))
    await asyncio.wait_for(a.prefilling.wait(), 5)
    await until(lambda: r.st["q27"] == "loaded")
    await until(lambda: mgr.active == 1)
    task.cancel()
    await until(lambda: mgr.active == 0 and all(not st.handoff_busy and st.inflight == 0 for st in p.app.pool.states.values()))
    a.pause.set()


@pytest.mark.parametrize("why", ["busy", "recent_demand", "min_mode"])
async def test_router_target_policy_blocks_switch(rig, why):
    policy = {"min_mode_s": 3600} if why == "min_mode" else None
    p, a, r, mgr = await rig("m35", policy=policy)
    if why == "busy":
        mgr.active = 1
    elif why == "recent_demand":
        mgr.record("q35"), mgr.record("q35")
    status, backend, reason, _ = await p.chat(U)
    mgr.active = 0
    assert backend == "3090"
    assert not ops(r, "load", "unload")
    assert not a.events      # recusa antes de gastar o prefill da origem
    assert p.app.metrics.handoffs["model_unavailable"] == 1


async def test_router_target_wrong_model(rig):
    p, a, r, mgr = await rig("m27", handoff_models=("pool", "other"))
    status, backend, reason, _ = await p.chat(U, body={"model": "other"})
    assert reason != "handoff"
    assert p.app.metrics.handoffs["wrong_model"] == 1
    assert not a.events and not any(e[0] in ("props", "restore") for e in r.events)


@pytest.mark.parametrize("what", ["gguf", "kv", "unified", "restore"])
async def test_router_target_identity_and_contract(rig, what):
    p, a, r, mgr = await rig("m27")
    if what == "gguf":
        r.gguf = "/models/35b.gguf"
    elif what == "kv":
        r.preset["cache-type-v"] = "q4_0"
    elif what == "unified":
        del r.preset["kv-unified"]    # n_stream do arquivo (np=1) nao casa com o filho nao unificado de np=2
    else:
        r.fail_restore = True
    status, backend, _, _ = await p.chat(U)
    assert status == 200 and backend == "3090"
    assert p.app.metrics.handoffs["fallback"] == 1
    assert not list(r.directory.iterdir()) and mgr.active == 0


def test_config_router_target_validation():
    base = [{"name": "s", "url": "http://x", "handoff_model_path": "/m", "handoff_gguf_path": "/m",
             "handoff_state_format": "f"},
            {"name": "t", "url": "http://y", "router": True, "modes": {"a": ["q27"]},
             "model_engine_ids": {"q27": "router-q27"}, "model_engine_owners": {"q27": "pool-proxy"},
             "handoff_model_path": "/m",
             "handoff_gguf_path": "/m", "handoff_state_format": "f", "handoff_model": "q27"}]
    settings = {"handoff_enabled": True, "handoff_source": "s", "handoff_target": "t", "handoff_models": ["pool"],
                "handoff_source_dir": "/a", "handoff_target_dir": "/b"}
    assert from_dict({"backends": base, "settings": settings}).backends[1].handoff_model == "q27"
    bad = [dict(base[0]), {**base[1], "handoff_model": "nope"}]
    with pytest.raises(ValueError, match="handoff_model"):
        from_dict({"backends": bad, "settings": settings})
    with pytest.raises(ValueError, match="origem direta"):
        from_dict({"backends": base, "settings": {**settings, "handoff_source": "t", "handoff_target": "s"}})


def test_deploy_example_loads():
    import os
    from pool_proxy import config as cfgmod
    cfg = cfgmod.load(os.path.join(os.path.dirname(__file__), "..", "deploy", "config-handoff-router.yaml"))
    target = cfg.backends[1]
    assert cfg.settings.handoff_enabled and target.router and target.handoff_model in target.modes["m27"]
    assert cfg.settings.handoff_source_dir == cfg.settings.handoff_target_dir
