"""ModeManager: arranjos de modelos ("modos") de um backend llama-server em router mode.

Um modo e um conjunto de modelos carregados ao mesmo tempo. A troca e decidida por demanda
(janela deslizante por modelo pedido) com histerese e executada em passos: drenar, salvar slots,
descarregar, carregar, restaurar, reabrir. Falha => volta ao modo anterior; sem sucesso no
retorno o backend fica unhealthy e uma tarefa tenta recuperar (nunca fica sem modelo em silencio).
"""
from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from collections import deque
from typing import Callable

from .config import BackendCfg, ModePolicy
from .routerctl import RUNNING, RouterCtl, RouterError, slot_filename
from .slotops import SlotOperationError

log = logging.getLogger("pool_proxy.modes")


class ModelUnavailable(Exception):
    """Modelo nao ficou disponivel dentro de switch_timeout_s (ou nao pertence ao backend)."""


class DrainTimeout(Exception):
    pass


class _LifecycleBlocked(Exception):
    """An accepted slot operation or unreadable identity ledger is unresolved."""


class _Step(Exception):
    def __init__(self, step: str, cause: Exception) -> None:
        super().__init__(f"{step}: {cause}")
        self.step = step
        self.cause = cause


class Lease:
    def __init__(self, mgr: "ModeManager") -> None:
        self._mgr = mgr
        self._done = False

    def release(self) -> None:
        if not self._done:
            self._done = True
            self._mgr._release()


class ModeManager:
    def __init__(self, cfg: BackendCfg, ctl: RouterCtl, metrics=None,
                 on_health: Callable[[bool], None] | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.cfg = cfg
        self.name = cfg.name
        self.pol: ModePolicy = cfg.mode_policy
        self.ctl = ctl
        self.metrics = metrics
        self.on_health = on_health or (lambda ok: None)
        self.clock = clock
        self.modes: dict[str, list[str]] = {k: list(v) for k, v in cfg.modes.items()}
        self.known: set[str] = {m for v in self.modes.values() for m in v}
        self.aliases = dict(cfg.model_aliases)
        self.mode: str | None = None            # None ate o primeiro sync
        self.mode_since = clock()
        self.switching: str | None = None       # modo de destino durante a troca
        self.broken = False
        self.epoch = 0                          # cresce quando um modelo cai: estado de slot anterior nao vale
        self.active = 0                         # pedidos que passaram pelo portao e ainda nao terminaram
        self.swap_s = self.pol.swap_estimate_s  # EMA das duracoes medidas de troca
        self.saved: dict[tuple[str, int], str] = {}
        # entradas [t, modelo, esperando]: `esperando` vira False quando o pedido passa pelo portao
        # (ou desiste); so pedidos esperando contam como demanda estrangeira.
        self._demand: deque[list] = deque()
        # demanda da faixa low, mesmo formato; so troca o modo com o router sem uso de primeiro plano
        self._low_demand: deque[list] = deque()
        self._no_slots: set[str] = set(cfg.no_slot_save)   # sem estado de slot (config ou 501 aprendido)
        self._cond = asyncio.Condition()
        self._lock = asyncio.Lock()
        self._kick = asyncio.Event()
        self._next_try = 0.0
        self._fails = 0
        self._next_recover = 0.0
        self._next_alive = 0.0
        self._task: asyncio.Task | None = None
        self._lifecycle_target: str | None = None
        # callable opcional: True => nao trocar de modo agora (janela do handoff 35B aberta)
        self.block_switch = None
        self._set_gauge()

    # ---- consulta ----------------------------------------------------------
    def resolve(self, model: str | None) -> str | None:
        """Nome canonico servido por este backend, ou None se ele nao serve o pedido."""
        m = self.aliases.get(model or "", model or "")
        if m in self.known:
            return m
        return self.cfg.default_model or None

    def models_now(self) -> set[str]:
        return set(self.modes.get(self.mode or "", ()))

    def available(self, model: str) -> bool:
        return (not self.broken and self.switching is None and self.mode is not None
                and model in self.modes[self.mode])

    def penalty(self, model: str | None) -> float:
        """Segundos somados ao custo de rotear para este backend (0 se o modelo ja esta servivel)."""
        m = self.resolve(model)
        if m is None or self.available(m):
            return 0.0
        return self.swap_s

    def mode_for(self, model: str) -> str | None:
        return next((k for k, v in self.modes.items() if model in v), None)

    # ---- demanda e decisao -------------------------------------------------
    def record(self, model: str) -> list:
        ent = [self.clock(), model, True]
        self._demand.append(ent)
        self._trim()
        self._kick.set()
        return ent

    def record_low(self, model: str) -> list:
        """Demanda da faixa low por um modelo fora do modo atual. Nao entra em `waiting_foreign` (pedido low nao
        segura o portao) e so pesa em `decide` sem nenhum uso de primeiro plano do modo atual."""
        ent = [self.clock(), model, True]
        self._low_demand.append(ent)
        self._trim()
        self._kick.set()
        return ent

    def _trim(self) -> None:
        lim = self.clock() - self.pol.demand_window_s
        while self._demand and self._demand[0][0] < lim:
            self._demand.popleft()
        while self._low_demand and (self._low_demand[0][0] < lim or not self._low_demand[0][2]):
            self._low_demand.popleft()

    def decide(self) -> str | None:
        """Modo de destino se a politica de histerese permitir trocar agora, senao None."""
        if self.mode is None or self.broken or self.switching is not None:
            return None
        if self.block_switch is not None and self.block_switch():
            return None
        now, p = self.clock(), self.pol
        if now < self._next_try or now - self.mode_since < p.min_mode_s:
            return None
        self._trim()
        cur = set(self.modes[self.mode])
        cur_n = sum(1 for _, m, _w in self._demand if m in cur)    # atendido ou nao: o modo atual esta em uso
        best: tuple[int, float, str] | None = None
        for name, models in self.modes.items():
            if name == self.mode:
                continue
            foreign = set(models) - cur
            ts = [t for t, m, w in self._demand if w and m in foreign]     # so quem espera no portao
            # dwell: ha quanto tempo a demanda espera (do primeiro pedido ate agora); com ts[-1]-ts[0]
            # um pedido sozinho nunca trocava e estourava a fila
            if len(ts) < max(1, p.switch_min_requests) or now - ts[0] < p.switch_min_dwell_s:
                continue
            # maior demanda; empate => quem espera ha mais tempo (determinista, nao a ordem da lista)
            if best is None or (len(ts), -ts[0]) > (best[0], -best[1]):
                best = (len(ts), ts[0], name)
        if best is None:
            return self._decide_low(now, cur, cur_n)
        if self.active > 0:
            if cur_n >= p.busy_threshold_requests:
                return None     # modo atual nem ocioso nem abaixo do limiar
            if p.switch_margin_requests > 0 and best[0] < cur_n + p.switch_margin_requests:
                return None     # margem opcional: empate favorece o modo atual em uso
        return best[2]

    def _decide_low(self, now: float, cur: set, cur_n: int) -> str | None:
        """Troca pedida pela faixa low (opt-in `low_switch`): so com o modo atual sem nenhum pedido de primeiro plano
        na janela, nada em voo (nem low) e a demanda low esperando ha `low_switch_dwell_s`. Primeiro plano do modo
        que saiu volta pela regra normal; o hot swap do router mainline e barato (35B<->27B ~9 s)."""
        p = self.pol
        if not p.low_switch or cur_n > 0 or self.active > 0:
            return None
        best: tuple[int, float, str] | None = None
        for name, models in self.modes.items():
            if name == self.mode:
                continue
            foreign = set(models) - cur
            ts = [t for t, m, w in self._low_demand if w and m in foreign]
            if len(ts) < max(1, p.low_switch_min_requests) or now - ts[0] < p.low_switch_dwell_s:
                continue
            if best is None or (len(ts), -ts[0]) > (best[0], -best[1]):
                best = (len(ts), ts[0], name)
        return best[2] if best else None

    def low_switch_possible(self, model: str) -> bool:
        """O pedido low por `model` pode vir a ser atendido por troca de modo (politica ligada e modelo conhecido)."""
        return self.pol.low_switch and not self.broken and self.mode_for(model) is not None

    def allow_handoff_switch(self, model: str) -> bool:
        """Politica para o handoff pedir a troca para o modo de `model`: mesma histerese da demanda
        (backoff de falha, tempo minimo no modo, nada em voo, modo atual abaixo do limiar de uso),
        mas um unico pedido basta: o prefill longo evitado nas MI50 vale mais que a troca."""
        if self.mode is None or self.broken or self.switching is not None or self.mode_for(model) is None:
            return False
        now, p = self.clock(), self.pol
        if now < self._next_try or now - self.mode_since < p.min_mode_s or self.active > 0:
            return False
        self._trim()
        cur = set(self.modes[self.mode])
        return sum(1 for _, m, _w in self._demand if m in cur) < p.busy_threshold_requests

    def can_acquire_for_handoff(self, model: str) -> bool:
        """Previsao sincrona de `acquire_for_handoff`: modelo servivel, troca em curso para o modo dele
        ou troca permitida pela politica. Deixa o handoff recusar antes de gastar o prefill da origem."""
        if self.broken or self.mode is None:
            return False
        if self.available(model):
            return True
        if self.switching is not None:
            return self.switching == self.mode_for(model)
        return self.allow_handoff_switch(model)

    async def acquire_for_handoff(self, model: str, timeout: float) -> Lease:
        """Lease do modelo ja servivel, ou espera uma troca em curso para o modo dele, ou (se a politica
        permitir) dispara a troca. ModelUnavailable se nao der: o chamador cai no roteamento normal."""
        deadline = self.clock() + timeout
        if not self.available(model):
            if self.switching is None:
                if not self.allow_handoff_switch(model):
                    raise ModelUnavailable(f"{self.name}: politica nao permite trocar para {model}")
                # shield: o timeout/cancelamento do handoff nao pode interromper uma troca no meio
                if not await asyncio.shield(asyncio.ensure_future(self.switch(self.mode_for(model), "handoff"))):
                    raise ModelUnavailable(f"{self.name}: troca para {model} falhou")
        async with self._cond:
            while True:
                if self.available(model):
                    self.active += 1
                    return Lease(self)
                left = deadline - self.clock()
                if left <= 0 or self.switching is None and self.mode is not None and model not in self.modes[self.mode]:
                    raise ModelUnavailable(f"{self.name}: {model} indisponivel")
                try:
                    await asyncio.wait_for(self._cond.wait(), left)
                except asyncio.TimeoutError:
                    raise ModelUnavailable(f"{self.name}: {model} indisponivel") from None

    # ---- portao de despacho --------------------------------------------------
    async def enter(self, model: str) -> Lease:
        """Registra demanda e espera (ate switch_timeout_s) o modelo estar servivel; devolve lease."""
        ent = self.record(model)
        deadline = self.clock() + self.pol.switch_timeout_s
        try:
            async with self._cond:
                while True:
                    if self.available(model):
                        self.active += 1
                        return Lease(self)
                    left = deadline - self.clock()
                    if left <= 0:
                        raise ModelUnavailable(f"{self.name}: {model} indisponivel")
                    try:
                        await asyncio.wait_for(self._cond.wait(), left)
                    except asyncio.TimeoutError:
                        raise ModelUnavailable(f"{self.name}: {model} indisponivel") from None
        finally:
            ent[2] = False      # atendido, expirado ou cancelado: deixa de ser demanda esperando

    def waiting_foreign(self) -> bool:
        """Ha pedido esperando no portao por um modelo fora do modo atual (troca pendente)."""
        if self.mode is None:
            return True
        cur = set(self.modes[self.mode])
        return any(w and m not in cur for _t, m, w in self._demand)

    def try_enter_low(self, model: str) -> "Lease | None":
        """Lease para a faixa de baixa prioridade: nao registra demanda (nunca provoca troca) e so entra se o
        modelo ja esta servivel e ninguem espera troca."""
        if not self.available(model) or self.waiting_foreign():
            return None
        self.active += 1
        return Lease(self)

    def _release(self) -> None:
        self.active -= 1
        asyncio.get_running_loop().create_task(self._notify())

    async def _notify(self) -> None:
        async with self._cond:
            self._cond.notify_all()

    # ---- ciclo de vida -----------------------------------------------------
    def start(self) -> None:
        self._task = asyncio.get_running_loop().create_task(self._run())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass

    async def _run(self) -> None:
        try:
            await self.sync()
        except Exception:
            log.exception("%s: sync inicial", self.name)
        while True:
            try:
                await asyncio.wait_for(self._kick.wait(), self.pol.eval_interval_s)
            except asyncio.TimeoutError:
                pass
            self._kick.clear()
            try:
                await self.tick()
            except Exception:
                log.exception("%s: avaliacao de modo", self.name)

    async def tick(self) -> None:
        if self.broken or self.mode is None:
            if self.clock() >= self._next_recover:
                self._next_recover = self.clock() + self.pol.recover_interval_s
                await self.recover()
            return
        if await self._check_alive():
            return
        target = self.decide()
        if target:
            low = not any(w for _t, m, w in self._demand if m in set(self.modes[target]))
            await self.switch(target, "low" if low else "demand")

    async def _check_alive(self) -> bool:
        """Modelo do modo atual que saiu (crash do filho, ex. GGML_ASSERT) deixava o proxy mandando pedidos que o
        router recusa com 400 `model is not loaded`. Confere a cada `alive_check_s`; se caiu, marca unhealthy e a
        recuperacao recarrega o modo (com recover_interval_s entre tentativas). True se marcou."""
        now, p = self.clock(), self.pol
        if p.alive_check_s <= 0 or now < self._next_alive or self.switching is not None or self._lock.locked():
            return False
        self._next_alive = now + p.alive_check_s
        try:
            states = await self.ctl.list_models()
        except RouterError:
            return False          # router fora: o health check do backend ja cuida
        if self.mode is None or self.switching is not None or self.broken:
            return False
        dead = [m for m in self.modes[self.mode] if not (states.get(m) and states[m].running)]
        if not dead:
            return False
        why = ", ".join(f"{m} exit={states[m].exit_code}" if states.get(m) else m for m in dead)
        await self._mark_broken(self.mode, f"modelo do modo {self.mode} caiu: {why}")
        return True

    async def _set_mode(self, mode: str | None, switching: str | None = None) -> None:
        async with self._cond:
            if mode != self.mode:
                self.mode_since = self.clock()
            self.mode, self.switching = mode, switching
            self._cond.notify_all()
        self._set_gauge()

    def _set_gauge(self) -> None:
        if self.metrics is not None:
            self.metrics.set_mode(self.name, self.mode or "", self.switching is not None, self.broken)

    # ---- sincronizacao inicial / recuperacao -------------------------------
    async def sync(self) -> None:
        """Alinha o router a um modo: o que ja corresponde ao conjunto carregado, senao o default."""
        await self._set_mode(self.mode, self.mode or self.cfg.default_mode)   # fecha o portao
        try:
            states = await self.ctl.list_models()
            running = {n for n, s in states.items() if s.running and n in self.known}
            mode = next((k for k, v in self.modes.items() if set(v) == running), None) \
                or self.mode or self.cfg.default_mode
            await self._apply(mode, states)
        except (RouterError, _Step, SlotOperationError, _LifecycleBlocked) as e:
            await self._mark_broken(self.mode or self.cfg.default_mode, f"sync: {e}")
            return
        try:
            await self._ensure_lifecycle_clear()
        except _LifecycleBlocked as e:
            await self._mark_broken(mode, f"sync publish gate: {e}")
            return
        await self._set_mode(mode)
        self._healthy()

    async def _apply(self, mode: str, states=None) -> None:
        """Descarrega os modelos conhecidos fora do modo e carrega os que faltam (um por vez)."""
        await self._ensure_lifecycle_clear()
        states = states or await self.ctl.list_models()
        want = list(self.modes[mode])
        for m in sorted(self.known - set(want)):
            if states.get(m) and states[m].running:
                await self._unload(m)
        for m in want:
            if not (states.get(m) and states[m].running):
                await self._load(m)
            else:
                await self.ctl.wait_loaded(m, self.pol.load_timeout_s)

    async def recover(self) -> None:
        if self._lifecycle_target is not None:
            # Keep the old mode and gate closed until its exact accepted operation
            # is terminal/ACKed or fenced by the authenticated same-engine owner.
            await self.switch(self._lifecycle_target, "slot-operation-reconcile")
            return
        if self.mode is None:
            await self.sync()
            return
        async with self._lock:
            target = self.mode
            await self._set_mode(target, target)
            try:
                await self._apply(target)
                await self._restore(set(self.modes[target]))
                await self._ensure_lifecycle_clear()
            except (RouterError, _Step, SlotOperationError, _LifecycleBlocked) as e:
                self.broken = True
                await self._set_mode(target)
                log.warning("%s: recuperacao falhou: %s", self.name, e)
                return
            self.broken = False
            await self._set_mode(target)
            self._healthy()
            log.info("%s: recuperado no modo %s", self.name, target)

    # ---- troca -------------------------------------------------------------
    async def switch(self, target: str, reason: str = "manual") -> bool:
        async with self._lock:
            frm = self.mode
            if frm is None or target == frm or target not in self.modes:
                return False
            t0 = self.clock()
            old, new = set(self.modes[frm]), set(self.modes[target])
            await self._set_mode(frm, target)            # fecha o portao: novos pedidos esperam
            log.info("%s: troca %s -> %s (%s)", self.name, frm, target, reason)
            try:
                await self._drain()
            except DrainTimeout:
                await self._fail(frm, target, "drain", t0)
                await self._set_mode(frm)
                return False
            except asyncio.CancelledError:
                self._hold_lifecycle_target(target, "cancelled during drain")
                raise
            try:
                await self._ensure_lifecycle_clear()
                if self.pol.save_slots:
                    await self._save(old - new)
                await self._ensure_lifecycle_clear()
                states = await self.ctl.list_models()
                for m in sorted(old - new):
                    if states.get(m) and states[m].running:
                        await self._unload(m)
                for m in self.modes[target]:
                    if m not in old or not (states.get(m) and states[m].running):
                        await self._load(m)
                await self._restore(new - old)
                # Last gate immediately before publishing the new mode/healthy state.
                await self._ensure_lifecycle_clear()
            except _LifecycleBlocked as e:
                self._hold_lifecycle_target(target, str(e))
                log.warning("%s: transicao mantida fechada; operacao de slot sem resolucao: %s", self.name, e)
                return False
            except asyncio.CancelledError:
                self._hold_lifecycle_target(target, "cancelled during transition")
                raise
            except (_Step, RouterError) as e:
                step = e.step if isinstance(e, _Step) else "router"
                log.error("%s: troca %s -> %s falhou (%s); voltando", self.name, frm, target, e)
                await self._fail(frm, target, step, t0)
                await self._rollback(frm, target)
                return False
            dt = self.clock() - t0
            self.swap_s = 0.7 * self.swap_s + 0.3 * dt
            self._next_try, self._fails = self.clock(), 0
            gone = old - new     # demanda velha dos modelos que sairam nao pesa na proxima decisao
            # mas quem ainda esta esperando no portao (w=True) nao pode ser descartado: sem a
            # demanda, decide() nunca pede a volta e o pedido expira em switch_timeout_s (503 na borda).
            self._demand = deque(x for x in self._demand if x[1] not in gone or x[2])
            await self._set_mode(target)
            self._lifecycle_target = None
            self._healthy()
            if self.metrics is not None:
                self.metrics.mode_switch(self.name, frm, target, "ok", dt)
            return True

    def _hold_lifecycle_target(self, target: str, reason: str) -> None:
        """Keep admission closed and let the automatic worker reconcile/resume."""
        self._lifecycle_target = target
        self.broken = True
        self._next_recover = self.clock() + self.pol.recover_interval_s
        self.on_health(False)
        self._set_gauge()
        log.warning("%s: transition to %s remains gated: %s", self.name, target, reason)

    async def _drain(self) -> None:
        deadline = self.clock() + self.pol.drain_timeout_s
        async with self._cond:
            while self.active > 0:
                left = deadline - self.clock()
                if left <= 0:
                    raise DrainTimeout()
                try:
                    await asyncio.wait_for(self._cond.wait(), min(left, 1.0))
                except asyncio.TimeoutError:
                    pass

    async def _fail(self, frm: str, to: str, step: str, t0: float) -> None:
        self._fails += 1
        self._next_try = self.clock() + self.pol.fail_backoff_s * 2 ** min(self._fails - 1, 6)
        gone = set(self.modes[to]) - set(self.modes[frm])
        self._demand = deque(x for x in self._demand if x[1] not in gone or x[2])  # exige demanda nova (quem espera fica)
        if self.metrics is not None:
            self.metrics.mode_switch(self.name, frm, to, "failed", self.clock() - t0, step)

    async def _rollback(self, frm: str, to: str) -> None:
        """Descarrega o que sobrou do destino, recarrega o modo anterior, restaura os slots."""
        try:
            await self._ensure_lifecycle_clear()
            states = await self.ctl.list_models()
            for m in sorted(set(self.modes[to]) - set(self.modes[frm])):
                if states.get(m) and states[m].running:
                    await self._unload(m)
            for m in self.modes[frm]:
                if not (states.get(m) and states[m].running):
                    await self._load(m)
            await self._restore(set(self.modes[frm]))
            await self._ensure_lifecycle_clear()
        except (_Step, RouterError, SlotOperationError, _LifecycleBlocked) as e:
            await self._mark_broken(frm, f"rollback: {e}")
            return
        await self._set_mode(frm)
        if self.metrics is not None:
            self.metrics.mode_switch(self.name, frm, to, "rolled_back", 0.0)

    async def _mark_broken(self, mode: str, why: str) -> None:
        log.error("%s: backend sem modelo servivel (%s); unhealthy ate recuperar", self.name, why)
        self.broken = True
        self.epoch += 1
        self._next_recover = self.clock() + self.pol.recover_interval_s
        if self.metrics is not None:
            self.metrics.mode_switch(self.name, mode, mode, "unhealthy", 0.0, "rollback")
        await self._set_mode(self.mode or mode)
        self.on_health(False)

    def _healthy(self) -> None:
        self.broken = False
        self._set_gauge()
        self.on_health(True)

    # ---- passos ------------------------------------------------------------
    async def _unload(self, model: str) -> None:
        try:
            await self.ctl.unload(model)
            await self.ctl.wait_unloaded(model, self.pol.unload_timeout_s)
        except RouterError as e:
            raise _Step("unload", e) from None

    async def _load(self, model: str) -> None:
        try:
            await self.ctl.load(model)
            await self.ctl.wait_loaded(model, self.pol.load_timeout_s)
        except RouterError as e:
            raise _Step("load", e) from None

    async def _save(self, models: set[str]) -> None:
        """Salva (best effort) os slots com cache relevante dos modelos que vao sair."""
        await self._ensure_lifecycle_clear()
        try:
            states = await self.ctl.list_models()
        except RouterError:
            return
        for m in sorted(models):
            if m in self._no_slots or not (states.get(m) and states[m].status in ("loaded",)):
                continue
            try:
                slots = await self.ctl.slots(m)
            except RouterError as e:
                self._slot_op("list", "error")
                log.warning("%s: /slots de %s falhou: %s", self.name, m, e)
                continue
            for s in slots:
                if m in self._no_slots:
                    break
                n = max(int(s.get("n_prompt_tokens") or 0), int(s.get("n_cached_tokens") or 0))
                if n < self.pol.save_min_tokens:
                    continue
                # Keep the router's established per-model/slot filename. `saved` is published only
                # after the terminal operation envelope has been validated.
                fn = slot_filename(m, int(s["id"]))
                meta = {"kind": "mode", "model": m, "slot": int(s["id"]), "filename": fn}
                def publish(envelope, model=m, sid=int(s["id"]), name=fn):
                    if envelope.get("status") == "completed":
                        self.saved[(model, sid)] = name
                try:
                    await self.ctl.slot_save(m, int(s["id"]), fn, publish=publish, metadata=meta)
                    self._slot_op("save", "ok")
                except (RouterError, SlotOperationError) as e:
                    if e.status == 501 and isinstance(e, SlotOperationError) \
                            and e.outcome == "preaccept_rejected":
                        self._no_state(m, "save")
                        break
                    if isinstance(e, SlotOperationError) and e.outcome in ("preaccept_rejected", "terminal_failed"):
                        self._slot_op("save", "error")
                        log.warning("%s: save %s slot %s rejeitado/terminal: %s", self.name, m, s["id"], e)
                        continue
                    self._slot_op("save", "error")
                    raise _LifecycleBlocked(f"save {m}/{s['id']}: {e}") from e

    async def _restore(self, models: set[str]) -> None:
        await self._ensure_lifecycle_clear()
        for (m, sid), fn in sorted(self.saved.items()):
            if m not in models:
                continue
            if m in self._no_slots:
                continue
            try:
                await self.ctl.slot_restore(m, sid, fn)
                self._slot_op("restore", "ok")
            except (RouterError, SlotOperationError) as e:
                if e.status == 501 and isinstance(e, SlotOperationError) \
                        and e.outcome == "preaccept_rejected":
                    self._no_state(m, "restore")
                    continue
                if isinstance(e, SlotOperationError) and e.outcome in ("preaccept_rejected", "terminal_failed"):
                    self._slot_op("restore", "error")
                    log.warning("%s: restore %s slot %s rejeitado/terminal: %s", self.name, m, sid, e)
                    continue
                else:
                    self._slot_op("restore", "error")
                    raise _LifecycleBlocked(f"restore {m}/{sid}: {e}") from e
            self.ctl.slot_operations_for(m).remove_index("mode", f"{m}:{sid}")
            self.saved.pop((m, sid), None)

    async def _ensure_lifecycle_clear(self) -> None:
        """Reconcile every persisted child ledger before any model state transition."""
        try:
            models = self.ctl.pending_operation_models()
            for model in sorted(models):
                client = self.ctl.slot_operations_for(model)
                timeout = max(0.001, min(client.timeout_s, 1.0))
                await asyncio.wait_for(client.reconcile_startup(
                    self.ctl.slotop_request(model), publish=self.recover_operation,
                    timeout_s=timeout), timeout=timeout)
        except (RouterError, SlotOperationError, sqlite3.Error, OSError, ValueError,
                asyncio.TimeoutError) as e:
            raise _LifecycleBlocked(str(e)) from e

    def recover_operation(self, metadata, envelope) -> None:
        if metadata.get("kind") == "mode" and envelope.get("status") == "completed":
            self.saved[(metadata["model"], int(metadata["slot"]))] = metadata["filename"]

    def _no_state(self, model: str, op: str) -> None:
        """501 = modelo sem `--slot-save-path`: sem estado a preservar, nao e erro; nao insiste mais."""
        if model not in self._no_slots:
            log.info("%s: %s sem suporte a slot (501); save/restore desligado para ele", self.name, model)
            self._no_slots.add(model)
        self._slot_op(op, "unsupported")

    def _slot_op(self, op: str, result: str) -> None:
        if self.metrics is not None:
            self.metrics.slot_op(self.name, op, result)
