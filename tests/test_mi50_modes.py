"""Config de producao das MI50 (deploy/config-mi50-router.yaml) contra um router falso."""
import os

import pytest

from conftest import Proxy, make_cfg
from pool_proxy import config as cfgmod
from pool_proxy.app import App
from pool_proxy.routerctl import slot_filename
from test_modes import FakeRouter, POLICY, until

YAML = os.path.join(os.path.dirname(__file__), "..", "deploy", "config-mi50-router.yaml")
MODELS = ("mi50-35b", "mi50-27b", "mineru-a")
U = [{"role": "user", "content": "oi"}]


def test_mi50_config_loads_and_is_safe():
    cfg = cfgmod.load(YAML)
    b = cfg.backends[0]
    assert b.router and b.default_mode == "m35"
    assert set(b.modes) == {"m35", "m27", "mineru1"}
    assert b.model_aliases["mineru"] == "mineru-a"
    # MI50: o unload espera pelo menos o stop-timeout (120 s) do preset
    assert b.mode_policy.unload_timeout_s >= 120
    assert b.mode_policy.fail_backoff_s >= 60
    ini = open(os.path.join(os.path.dirname(__file__), "rtmi50.ini"), encoding="utf-8").read()
    assert "stop-timeout = 120" in ini
    for m in {m for v in b.modes.values() for m in v}:
        assert f"[{m}]" in ini


@pytest.fixture
async def mi50(tmp_path):
    r = FakeRouter(models=MODELS)
    await r.start()
    raw = cfgmod.load(YAML).backends[0]
    d = {"name": "mi50-router", "url": r.url, "np": raw.np, "router": True, "default_mode": raw.default_mode,
         "default_model": raw.default_model, "model_aliases": raw.model_aliases, "modes": raw.modes,
         "model_engine_ids": {m: "router-" + m for group in raw.modes.values() for m in group},
         "model_engine_owners": {m: "pool-proxy" for group in raw.modes.values() for m in group},
         "slotops_ledger_dir": str(tmp_path / "router-ledger"),
         "mode_policy": {**POLICY, "switch_min_requests": 1}}
    p = await Proxy(App(make_cfg([d], models=["qwen3.6-35b", "qwen3.8-27b", "mineru"], queue_timeout_s=5))).start()
    mgr = p.app.modes["mi50-router"]
    await until(lambda: mgr.mode == "m35" and mgr.switching is None)
    yield r, p, mgr
    await p.stop()
    await r.stop()


async def test_three_mode_cycle_with_aliases_and_slot_restore(mi50):
    r, p, mgr = mi50
    assert r.st == {"mi50-35b": "loaded", "mi50-27b": "unloaded", "mineru-a": "unloaded"}
    r.slots["mi50-35b"] = [{"id": 0, "n_prompt_tokens": 3000}, {"id": 1, "n_prompt_tokens": 5}]
    st, _, _, txt = await p.chat(U, body={"model": "qwen3.8-27b"})
    assert st == 200 and mgr.mode == "m27" and '"mi50-27b"' in txt
    assert r.st == {"mi50-35b": "unloaded", "mi50-27b": "loaded", "mineru-a": "unloaded"}
    st, _, _, txt = await p.chat(U, body={"model": "mineru"})
    assert st == 200 and mgr.mode == "mineru1" and '"mineru-a"' in txt
    st, _, _, txt = await p.chat(U, body={"model": "qwen3.6-35b"})
    assert st == 200 and mgr.mode == "m35"
    # save so do slot com cache, na saida do 35B; restore na volta; nunca dois modelos de GPU ao mesmo tempo
    fn = slot_filename("mi50-35b", 0)
    assert ("save", "mi50-35b", 0, fn, "false") in r.calls and ("restore", "mi50-35b", 0, fn, "false") in r.calls
    loaded = set()
    for c in r.calls:
        if c[0] == "load":
            loaded.add(c[1])
        elif c[0] == "unload":
            loaded.discard(c[1])
        assert len(loaded) <= 1, c
