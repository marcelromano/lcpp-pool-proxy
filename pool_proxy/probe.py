"""SlotProbe real: afinidade de prefixo via telemetria slot-telemetry do llama-server.

Fluxo por requisicao:
  1. /slots de cada backend (cache curto, timeout baixo; falha/timeout => backend ignorado no probe).
  2. Tokenizacao do prompt renderizado em UM backend com slot-telemetry, via
     POST /v1/chat/completions/input_tokens {return_ids:true} (1 chamada) ou, se o endpoint nao
     devolver prefix_hash, POST /apply-template + POST /tokenize {return_ids:true} (2 chamadas).
     O hash vem do servidor (FNV-1a 64 cumulativo por bloco de 256), igual ao do /slots.
     Resultado em LRU por hash do corpo (messages/tools/kwargs).
  3. Para cada slot: prefixo reaproveitavel (ver `usable_prefix`). Backend = melhor slot.

Regra de prefixo utilizavel (modelo hibrido):
  k      = blocos iniciais iguais entre prefix_hash do pedido e do slot; base = k * block.
  - Se o pedido ESTENDE todo o cache do slot (todos os blocos completos do slot iguais, o slot tem
    >= 1 bloco e n_prompt >= n_cached_tokens): usa n_cached_tokens (estado vivo; a cauda parcial
    apos o ultimo bloco nao e verificavel por hash e e assumida igual).
  - Senao: usable = base; se n_checkpoints > 0, usable = min(base, ckpt_pos_max + 1)
    (so ha estado restauravel ate o ultimo checkpoint; sem checkpoint valido => 0).
  - Sempre limitado a n_prompt - 1 (o servidor reprocessa ao menos 1 token).
  Com has_media o hash cobre so o texto: a estimativa e do trecho de texto.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import aiohttp

log = logging.getLogger("pool_proxy.probe")

TEMPLATE_KEYS = ("messages", "tools", "tool_choice", "chat_template_kwargs")


@dataclass
class Affinity:
    reuse: int = 0
    slot: int | None = None


@dataclass
class ProbeResult:
    prompt_tokens: int | None = None
    backends: dict[str, Affinity] = field(default_factory=dict)   # so backends que responderam (slot-telemetry)


def usable_prefix(slot: dict, req_hash: list[str], n_prompt: int, block: int = 256) -> int:
    sh = slot.get("prefix_hash") or []
    k = 0
    for a, b in zip(sh, req_hash):
        if a != b:
            break
        k += 1
    base = k * block
    n_cached = int(slot.get("n_cached_tokens") or 0)
    if sh and k == len(sh) and n_prompt >= n_cached:
        usable = n_cached
    else:
        usable = base
        if int(slot.get("n_checkpoints") or 0) > 0:
            usable = min(base, max(0, int(slot.get("ckpt_pos_max", -1)) + 1))
    return max(0, min(usable, n_prompt - 1))


class ServerSlotProbe:
    def __init__(self, settings) -> None:
        self.s = settings
        self.session: aiohttp.ClientSession | None = None
        self.auth: Callable[[Any], dict] = lambda st: {}
        self.metrics = None
        self.caps: dict[str, bool] = {}            # nome -> tem slot-telemetry
        self._cap_checked: dict[str, float] = {}
        self._slots_cache: dict[str, tuple[float, list]] = {}
        self._slots_tasks: dict[str, asyncio.Task] = {}
        self._tok_cache: OrderedDict[str, dict] = OrderedDict()
        self._tok_mode: dict[str, str] = {}        # nome -> "input_tokens" | "apply_tokenize"

    def attach(self, session, auth, metrics) -> None:
        self.session, self.auth, self.metrics = session, auth, metrics

    # ---- util ------------------------------------------------------------
    def _obs(self, stage: str, t0: float, backend: str = "", err: bool = False) -> None:
        if self.metrics:
            self.metrics.probe(stage, time.monotonic() - t0, backend, err)

    async def _post(self, st, path: str, body: dict, timeout: float):
        async with self.session.post(
                st.cfg.url.rstrip("/") + path, json=body, headers=self.auth(st),
                timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            if r.status != 200:
                await r.read()
                return r.status, None
            return r.status, json.loads(await r.read())

    # ---- /slots ----------------------------------------------------------
    async def _fetch_slots(self, st) -> list | None:
        name, t0 = st.cfg.name, time.monotonic()
        try:
            async with self.session.get(
                    st.cfg.url.rstrip("/") + "/slots", headers=self.auth(st),
                    timeout=aiohttp.ClientTimeout(total=self.s.probe_slots_timeout_s)) as r:
                if r.status != 200:
                    await r.read()
                    self.caps[name], self._cap_checked[name] = False, time.monotonic()
                    self._obs("slots", t0, name)
                    return None
                data = json.loads(await r.read())
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError, ValueError):
            self._obs("slots", t0, name, err=True)
            return None      # timeout/erro: ignora no probe, nao altera a capability conhecida
        self._obs("slots", t0, name)
        if not isinstance(data, list):
            data = data.get("slots", []) if isinstance(data, dict) else []
        self.caps[name] = any(isinstance(s, dict) and "prefix_hash" in s for s in data)
        self._cap_checked[name] = time.monotonic()
        if not self.caps[name]:
            return None
        self._slots_cache[name] = (time.monotonic(), data)
        return data

    async def slots(self, st) -> list | None:
        name = st.cfg.name
        if self.caps.get(name) is False and \
                time.monotonic() - self._cap_checked.get(name, 0) < self.s.probe_recheck_s:
            return None
        hit = self._slots_cache.get(name)
        if hit and time.monotonic() - hit[0] < self.s.probe_slots_ttl_s:
            return hit[1]
        task = self._slots_tasks.get(name)
        if task is None or task.done():
            task = asyncio.create_task(self._fetch_slots(st))
            self._slots_tasks[name] = task
        return await asyncio.shield(task)

    async def detect(self, backends: Sequence[Any]) -> None:
        await asyncio.gather(*(self.slots(st) for st in backends if not st.cfg.overflow),
                             return_exceptions=True)

    # ---- tokenizacao -----------------------------------------------------
    async def _tokenize(self, st, body: dict) -> dict | None:
        """Contagem (e, se o engine expuser slot-telemetry, hash) do prompt renderizado.

        `n` volta sempre que o engine responde a contagem, MESMO SEM prefix_hash: e o numero exato de tokens
        do prompt e substitui a estimativa chars/chars_per_token na admissao ( — nenhum dos dois engines
        de producao expoe prefix_hash, entao a estimativa era usada em 100% dos pedidos).
        `hash` None => sem dado de reuso de prefixo; so a contagem e utilizavel.
        """
        tb = {k: body[k] for k in TEMPLATE_KEYS if k in body}
        key = hashlib.sha256(json.dumps(tb, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if key in self._tok_cache:
            self._tok_cache.move_to_end(key)
            return self._tok_cache[key]
        name, t0, res = st.cfg.name, time.monotonic(), None
        to = self.s.probe_tokenize_timeout_s
        try:
            if self._tok_mode.get(name) != "apply_tokenize":
                _, d = await self._post(st, "/v1/chat/completions/input_tokens",
                                        {**tb, "return_ids": True}, to)
                n = int(d.get("input_tokens") or len(d.get("ids") or [])) if isinstance(d, dict) else 0
                if n > 0:
                    self._tok_mode[name] = "input_tokens"
                    res = {"n": n, "hash": d.get("prefix_hash"),
                           "block": int(d.get("prefix_hash_block", 256) or 256)}
                else:
                    self._tok_mode[name] = "apply_tokenize"
            if res is None:
                _, a = await self._post(st, "/apply-template", tb, to)
                if a and isinstance(a.get("prompt"), str):
                    # mesma tokenizacao do chat: add_special=true, parse_special=true
                    _, d = await self._post(st, "/tokenize",
                                            {"content": a["prompt"], "add_special": True,
                                             "parse_special": True, "return_ids": True}, to)
                    if isinstance(d, dict):
                        ids = d.get("ids") or d.get("tokens") or []
                        n = int(d.get("n_tokens") or len(ids) or 0)
                        if n > 0:
                            res = {"n": n, "hash": d.get("prefix_hash"),
                                   "block": int(d.get("prefix_hash_block", 256) or 256)}
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError, ValueError):
            self._obs("tokenize", t0, name, err=True)
            return None
        self._obs("tokenize", t0, name, err=res is None)
        if res is not None:
            self._tok_cache[key] = res
            while len(self._tok_cache) > self.s.probe_cache_size:
                self._tok_cache.popitem(last=False)
        return res

    # ---- API -------------------------------------------------------------
    async def analyze(self, backends: Sequence[Any], body: dict) -> ProbeResult:
        out = ProbeResult()
        cands = [st for st in backends if not st.cfg.overflow and st.healthy]
        if not cands or self.session is None:
            return out
        got = await asyncio.gather(*(self.slots(st) for st in cands), return_exceptions=True)
        live = [(st, s) for st, s in zip(cands, got) if isinstance(s, list) and self.caps.get(st.cfg.name)]
        # A contagem exata de tokens NAO depende de slot-telemetry: sem backend "live" (nenhum expoe prefix_hash)
        # ainda tokenizamos um candidato, senao a admissao fica presa na estimativa chars/chars_per_token
        # (nenhum dos dois engines de producao expoe prefix_hash).
        tok = None
        for st in ([st for st, _ in live] or cands):   # um backend basta (mesmo modelo/tokenizador)
            tok = await self._tokenize(st, body)
            if tok:
                break
        if not tok:
            return out
        out.prompt_tokens = tok["n"]
        for st, slots in live:
            best, best_busy = Affinity(), False
            for sl in slots:
                if not isinstance(sl, dict):
                    continue
                u = usable_prefix(sl, tok.get("hash") or [], tok["n"], int(tok["block"]))
                busy = bool(sl.get("is_processing"))
                if u > best.reuse or (u == best.reuse and u > 0 and best_busy and not busy):
                    best, best_busy = Affinity(u, sl.get("id")), busy
            out.backends[st.cfg.name] = best
        return out

    async def find_owner(self, backends, messages):
        r = await self.analyze(backends, {"messages": messages})
        best = max(r.backends.items(), key=lambda kv: kv[1].reuse, default=None)
        return best[0] if best and best[1].reuse > 0 else None
