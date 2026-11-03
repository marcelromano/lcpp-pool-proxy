import pytest

from pool_proxy import config as cfgmod, paths, routerctl


@pytest.fixture
def no_state_env(monkeypatch):
    monkeypatch.delenv('POOL_PROXY_STATE_DIR', raising=False)
    return monkeypatch


def test_default_state_dir_is_var_lib(no_state_env):
    assert paths.state_dir() == '/var/lib/pool-proxy'
    assert paths.state_path('slot-operations') == '/var/lib/pool-proxy/slot-operations'


def test_env_overrides_state_dir_for_all_defaults(no_state_env, tmp_path):
    no_state_env.setenv('POOL_PROXY_STATE_DIR', str(tmp_path))
    assert cfgmod.BackendCfg(name='b', url='http://127.0.0.1:1').slotops_ledger_dir == str(tmp_path / 'slot-operations')
    assert cfgmod.Handoff35().ledger_dir == str(tmp_path / 'slot-operations')
    ctl = routerctl.RouterCtl(lambda: None, 'http://127.0.0.1:1')
    assert str(ctl.ledger_dir) == str(tmp_path / 'slot-operations')


def test_explicit_ledger_dir_wins_over_env(no_state_env, tmp_path):
    no_state_env.setenv('POOL_PROXY_STATE_DIR', str(tmp_path / 'env'))
    ctl = routerctl.RouterCtl(lambda: None, 'http://127.0.0.1:1', ledger_dir=tmp_path / 'explicito')
    assert ctl.ledger_dir == tmp_path / 'explicito'


def test_blank_env_falls_back_to_default(no_state_env):
    no_state_env.setenv('POOL_PROXY_STATE_DIR', '  ')
    assert paths.state_dir() == '/var/lib/pool-proxy'
