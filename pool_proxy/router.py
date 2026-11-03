from __future__ import annotations

import asyncio
import time
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass, field

from .config import BackendCfg, Config, Settings


class QueueTimeout(Exception):
    pass


@dataclass
class RequestInfo:
    messages: list[dict]
    est_prompt_tokens: int
    max_tokens: int
    has_image: bool
    chain: list[str] = field(default_factory=list)   # hash em cadeia: chain[k-1] = prefixo de k mensagens
    chain_est: list[int] = field(default_factory=list)  # tokens estimados do prefixo de k mensagens
    # preenchidos pelo SlotProbe (slot-telemetry): so backends que responderam ao probe
    reuse: dict[str, int] = field(default_factory=dict)       # prefixo reaproveitavel (tokens)
    slot: dict[str, int | None] = field(default_factory=dict) # id_slot de maior afinidade
    exact_tokens: bool = False
    hint: str | None = None          # probe legado (find_owner): backend dono do prefixo
    model: str = ""                  # campo `model` do pedido (usado por backends router)

    @classmethod
    def from_body(cls, body: dict, s: Settings) -> "RequestInfo":
        msgs = body.get("messages") or []
        chars, imgs = 0, 0
        for m in msgs:
            c = m.get("content")
            if isinstance(c, str):
                chars += len(c)
            elif isinstance(c, list):
                for p in c:
                    if not isinstance(p, dict):
                        continue
                    if p.get("type") in ("image_url", "image", "input_image"):
                        imgs += 1
                    elif isinstance(p.get("text"), str):
                        chars += len(p["text"])
            if m.get("tool_calls"):
                chars += len(json.dumps(m["tool_calls"]))
        if body.get("tools"):
            chars += len(json.dumps(body["tools"]))
        est = int(chars / s.chars_per_token) + imgs * s.image_tokens
        mt = body.get("max_tokens") or body.get("max_completion_tokens") or s.default_max_tokens
        chain, h, cest, acc = [], hashlib.sha256(), [], 0
        for m in msgs:
            raw = json.dumps(m, sort_keys=True, ensure_ascii=False)
            h.update(raw.encode())
            chain.append(h.copy().hexdigest())
            acc += len(raw)
            cest.append(int(acc / s.chars_per_token))
        mdl = body.get("model")
        return cls(msgs, est, int(mt), imgs > 0, chain, cest, model=mdl if isinstance(mdl, str) else "")


class BackendState:
    def __init__(self, cfg: BackendCfg, default_service_s: float) -> None:
        self.cfg = cfg
        self.sem = asyncio.Semaphore(max(1, cfg.np))
        self.inflight = 0
        self.queued = 0
        self.healthy = True
        self.fails = 0
        self.avg_s = default_service_s
        self.modes = None                # ModeManager, so em backend router
        self.handoff_busy = False
        self.idle_since = 0.0             # monotonic do fim do ultimo pedido de primeiro plano (faixa low)
        self.low_inflight = 0             # pedidos low em curso ou aguardando slot (contidos em inflight ao rodar)
        # sessao (hash ate a 2a mensagem) cujo KV ocupa cada slot, conhecida pelo proxy (faixa low fixa o slot)
        self.slot_sid: dict[int, str] = {}
        self.slot_used: dict[int, int] = {}      # ordem de uso (contador) para o LRU
        self.slot_seq = 0
        self.slot_epoch = 0                  # epoca do router quando slot_sid foi escrito
        self.slot_tokens: dict[int, int] = {}   # tokens estimados residentes por slot (faixa low)
        self.slot_dirty: dict[int, object] = {}  # slot -> RequestInfo do ultimo turno ainda nao salvo
        self.slot_busy: set[int] = set()     # slots reservados por pedidos low em curso

    def wait_est(self) -> float:
        ahead = max(0, self.inflight + self.queued - self.cfg.np + 1)
        return ahead * self.avg_s / max(1, self.cfg.np)

    def cost(self, info: RequestInfo, reuse: int = 0) -> float:
        """fila + (tokens_prompt - prefixo_reaproveitado)/prefill_tps + max_tokens/decode_tps."""
        c = self.cfg
        fresh = max(0, info.est_prompt_tokens - reuse)
        own = fresh / c.prefill_tps + info.max_tokens / c.decode_tps
        swap = self.modes.penalty(info.model) if self.modes is not None else 0.0
        return (self.wait_est() + own + swap) / max(c.weight, 1e-6)


class Pool:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.s = cfg.settings
        self.states: dict[str, BackendState] = {
            b.name: BackendState(b, self.s.default_service_s) for b in cfg.backends
        }
        self.overflow = next((st for st in self.states.values() if st.cfg.overflow), None)
        self.affinity: OrderedDict[str, str] = OrderedDict()

    # ---- afinidade -------------------------------------------------------
    def _affinity_lookup(self, info: RequestInfo) -> str | None:
        r = self._affinity_hit(info)
        return r[0] if r else None

    def _affinity_hit(self, info: RequestInfo) -> tuple[str, int] | None:
        # prefixo = mensagens exceto a ultima; tenta o mais longo primeiro
        for k in range(len(info.chain) - 1, 0, -1):
            name = self.affinity.get(info.chain[k - 1])
            if name:
                return name, k
        return None

    def record_affinity(self, info: RequestInfo, name: str) -> None:
        for k in (len(info.chain) - 1, len(info.chain)):
            if k >= 1:
                key = info.chain[k - 1]
                self.affinity[key] = name
                self.affinity.move_to_end(key)
        while len(self.affinity) > self.s.affinity_capacity:
            self.affinity.popitem(last=False)

    # ---- roteamento ------------------------------------------------------
    def _eligible(self, info: RequestInfo, exclude: set[str], overflow: bool = False) -> list[BackendState]:
        out = []
        for st in self.states.values():
            c = st.cfg
            if c.overflow != overflow or not st.healthy or c.name in exclude:
                continue
            if c.router and (st.modes is None or st.modes.resolve(info.model) is None):
                continue
            if not c.router and c.models and info.model not in c.models:
                continue
            if info.has_image and not c.vision:
                continue
            if c.ctx_max and info.est_prompt_tokens + info.max_tokens > c.ctx_max:
                continue
            out.append(st)
        return out

    def route(self, info: RequestInfo, exclude=frozenset(), hint: str | None = None):
        if hint:
            info.hint = hint
        """Escolhe backend e incrementa `queued` (sem await entre decisao e reserva)."""
        cands = self._eligible(info, set(exclude))
        decision = self._decide(info, cands, hint)
        if decision is None or (decision[0].wait_est() > self.s.spill_after_s and self.overflow):
            ov = self._eligible(info, set(exclude), overflow=True)
            if ov:
                decision = (ov[0], "spill" if cands else "no_backend")
        if decision is None:
            return None
        decision[0].queued += 1
        return decision

    def _decide(self, info, cands, hint):
        if not cands:
            return None
        byname = {st.cfg.name: st for st in cands}
        if info.reuse:
            return self._decide_probed(info, cands)
        # 2. afinidade (espelho local, depois sonda de /slots)
        for name in (self._affinity_lookup(info), info.hint):
            st = byname.get(name) if name else None
            if st and st.wait_est() <= self.s.affinity_max_wait_s:
                return st, "affinity"
        # 3. imagem: candidatos ja filtrados por vision
        tag = "vision" if info.has_image else "cost"
        # 4. prompt longo e frio -> maior prefill_tps
        if info.est_prompt_tokens >= self.s.long_prompt_tokens:
            ok = [st for st in cands if st.wait_est() <= self.s.spill_after_s] or cands
            best = max(ok, key=lambda st: (st.cfg.prefill_tps, -st.cost(info)))
            return best, "long_cold"
        # 5. menor custo estimado
        return min(cands, key=lambda st: st.cost(info)), tag

    def _decide_probed(self, info, cands):
        """Com dados slot-telemetry: menor custo com prefixo reaproveitado por backend. Backend sem slot-telemetry
        (ex.: ik) usa o espelho local como estimativa (tokens estimados do prefixo conhecido)."""
        mirror = self._affinity_hit(info)
        eff: dict[str, int] = {}
        for st in cands:
            n = st.cfg.name
            if n in info.reuse:
                eff[n] = info.reuse[n]
            elif mirror and mirror[0] == n:
                eff[n] = info.chain_est[mirror[1] - 1]
            else:
                eff[n] = 0
        best = min(cands, key=lambda st: st.cost(info, eff[st.cfg.name]))
        if eff[best.cfg.name] > 0:
            tag = "affinity"
        else:
            tag = "vision" if info.has_image else "cost"
        return best, tag

    def spill_target(self, info: RequestInfo, exclude=frozenset()):
        ov = self._eligible(info, set(exclude), overflow=True)
        if not ov:
            return None
        ov[0].queued += 1
        return ov[0], "spill_timeout"

    # ---- concorrencia / fila ---------------------------------------------
    async def acquire(self, st: BackendState) -> None:
        """Consome a reserva `queued` e espera um slot (limite np), com timeout."""
        try:
            await asyncio.wait_for(st.sem.acquire(), self.s.queue_timeout_s)
        except asyncio.TimeoutError:
            raise QueueTimeout(st.cfg.name) from None
        finally:
            st.queued -= 1
        st.inflight += 1

    def release(self, st: BackendState, seconds: float | None, low: bool = False) -> None:
        st.inflight -= 1
        st.sem.release()
        if low:
            st.low_inflight -= 1
        elif st.inflight - st.low_inflight <= 0:
            st.idle_since = time.monotonic()
        if seconds is not None:
            a = self.s.ema_alpha
            st.avg_s = (1 - a) * st.avg_s + a * seconds

    def low_reuse(self, info: RequestInfo) -> dict[str, int]:
        """Prefixo reaproveitavel estimado por backend elegivel: probe slot-telemetry quando houver, senao o espelho local."""
        mirror = self._affinity_hit(info)
        eff: dict[str, int] = {}
        for st in self._eligible(info, set()):
            n = st.cfg.name
            if n in info.reuse:
                eff[n] = info.reuse[n]
            elif mirror and mirror[0] == n:
                eff[n] = info.chain_est[mirror[1] - 1]
            else:
                eff[n] = 0
        return eff

    def low_eligible(self, info: RequestInfo, exclude=frozenset()) -> list[BackendState]:
        return self._eligible(info, set(exclude))

    def low_candidate(self, info: RequestInfo, now: float, since: float | None = None,
                      exclude=frozenset()) -> BackendState | None:
        """Faixa de baixa prioridade: backend saudavel, sem pedido de primeiro plano em curso nem fila, sem primeiro
        plano ha >= low_idle_s, sem handoff, e com vaga low (low_max_inflight, limitado a np); o de menor custo.
        Low seguido de low nao espera o vao: so o primeiro plano protege. Modo (router) e conferido pelo chamador."""
        elig = self._eligible(info, set(exclude))
        free = [st for st in elig
                if st.inflight - st.low_inflight <= 0 and st.queued == 0 and not st.handoff_busy
                and st.low_inflight < min(self.s.low_max_inflight, st.cfg.np)
                and now - st.idle_since >= self.s.low_idle_s]
        if not free:
            return None
        # Afinidade (2026-10-10): o worker de agente reenvia o contexto inteiro a cada turno; trocar de backend refaz o
        # prefill (nas MI50 ~200 t/s contra ~800 na 3090). Prefere quem ja tem o prefixo e espera o dono ocupado.
        eff = self.low_reuse(info)
        owner = max(elig, key=lambda st: eff.get(st.cfg.name, 0))
        if (since is not None and owner not in free and eff.get(owner.cfg.name, 0) >= self.s.low_sticky_tokens
                and now - since < self.s.low_sticky_wait_s):
            return None
        return max(free, key=lambda st: (eff.get(st.cfg.name, 0), -st.cost(info, eff.get(st.cfg.name, 0))))

    # ---- saude -----------------------------------------------------------
    def mark_failure(self, st: BackendState, hard: bool = False) -> None:
        st.fails += 1
        if hard or st.fails >= self.s.health_failures:
            st.healthy = False
            st.slot_sid.clear()          # engine caiu/reiniciou: o conteudo dos slots nao e mais conhecido
            st.slot_dirty.clear()
            st.slot_tokens.clear()

    def mark_ok(self, st: BackendState) -> None:
        st.fails = 0
        st.healthy = True
