"""Diretorio de estado do pool-proxy (ledgers de slot-operations, log de requests).

Ordem de resolucao: variavel POOL_PROXY_STATE_DIR (se definida e nao vazia), senao
/var/lib/pool-proxy. Caminhos sao resolvidos no momento da chamada, nao na importacao.
"""
import os

STATE_DIR_ENV = 'POOL_PROXY_STATE_DIR'
DEFAULT_STATE_DIR = '/var/lib/pool-proxy'


def state_dir() -> str:
    return os.environ.get(STATE_DIR_ENV, '').strip() or DEFAULT_STATE_DIR


def state_path(*parts: str) -> str:
    return os.path.join(state_dir(), *parts)
