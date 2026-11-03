"""Ganchos de extensao.

SlotProbe (implementado em probe.py): consulta o `prefix_hash` do /slots (patch slot-telemetry do
llama-server) para saber qual backend ja tem o prefixo da conversa em cache.
HandoffPlanner: ponto de chamada para handoff de KV via kvhandoff.py (prefill na 3090,
restore na MI50). Ainda nao implementado (NullHandoff).
"""
from __future__ import annotations

from typing import Any, Protocol, Sequence


class SlotProbe(Protocol):
    async def analyze(self, backends: Sequence[Any], body: dict) -> Any:
        """ProbeResult (pool_proxy.probe): tokens exatos do prompt e afinidade por backend."""


class HandoffPlanner(Protocol):
    async def before_forward(self, info: Any, backend: Any) -> None:
        """Chamado depois da escolha e antes do repasse (ex.: restore de KV no destino)."""


class NullSlotProbe:
    async def analyze(self, backends, body):
        from .probe import ProbeResult
        return ProbeResult()


class NullHandoff:
    async def before_forward(self, info, backend):
        return None
