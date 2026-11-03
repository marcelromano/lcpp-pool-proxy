from __future__ import annotations

import argparse
import hmac
import asyncio
import json
import logging
import os
import re
import time
import uuid

import aiohttp
from aiohttp import web

from . import config as cfgmod
from . import request_log
from . import schemas
from .hooks import HandoffPlanner, NullHandoff, NullSlotProbe, SlotProbe
from .probe import ProbeResult, ServerSlotProbe
from .metrics import Metrics
from .modes import ModelUnavailable, ModeManager
from .routerctl import RouterCtl
from .router import BackendState, Pool, QueueTimeout, RequestInfo

log = logging.getLogger("pool_proxy")

# Cabecalhos nao repassados a montante. X-Conversation-Id faria a engine ignorar a
# desconexao do cliente (stream retomavel) e continuar gerando.
DROP_REQ = {"host", "content-length", "authorization", "x-conversation-id", "connection",
            "transfer-encoding", "keep-alive", "te", "upgrade", "accept-encoding"}
DROP_RESP = {"content-length", "transfer-encoding", "connection", "keep-alive", "content-encoding"}


_CACHE_N = re.compile(rb'"cache_n"\s*:\s*(\d+)')


def _cache_n(tail: bytes) -> int | None:
    """Ultimo `timings.cache_n` do corpo/ultimo trecho SSE (so engines que o enviam)."""
    m = _CACHE_N.findall(tail)
    return int(m[-1]) if m else None


class UpstreamDown(Exception):
    pass


class KvCapacity(Exception):
    """503 de admissao do backend (`kv_capacity_error`): carrega a resposta original para devolver ao cliente."""

    def __init__(self, resp):
        super().__init__("kv_capacity_error")
        self.resp = resp


def _json_object_to_schema(raw: bytes, body: dict) -> tuple[bytes, dict]:
    """llama.cpp (mainline e ik) nao impoe `json_object` sem schema; `json_schema` e imposto. Converte para um
    schema generico de objeto, para o modelo nao devolver JSON dentro de cerca de markdown."""
    rf = body.get("response_format")
    if not isinstance(rf, dict) or rf.get("type") != "json_object" or rf.get("schema"):
        return raw, body
    body = {**body, "response_format": {"type": "json_schema",
                                        "json_schema": {"name": "json_object", "schema": {"type": "object"}}}}
    return json.dumps(body, ensure_ascii=False).encode(), body


def _auth(st: BackendState) -> dict[str, str]:
    key = os.environ.get(st.cfg.key_env, "") if st.cfg.key_env else ""
    return {"Authorization": f"Bearer {key}"} if key else {}


class App:
    def __init__(self, cfg: cfgmod.Config, probe: SlotProbe | None = None,
                 handoff: HandoffPlanner | None = None) -> None:
        self.cfg = cfg
        request_log.start()
        self.pool = Pool(cfg)
        self.metrics = Metrics()
        self.probe = probe or (ServerSlotProbe(cfg.settings) if cfg.settings.probe_enabled
                               else NullSlotProbe())
        self.handoff = handoff or NullHandoff()
        from .handoff import ColdHandoff
        self.cold_handoff = ColdHandoff(self)
        from .swap35 import Swap35
        self.swap35 = Swap35(self)
        from .sessions import LowSessions
        self.sessions = LowSessions(self)
        self.session: aiohttp.ClientSession | None = None
        self._health_task: asyncio.Task | None = None
        self.modes: dict[str, ModeManager] = {}
        self._started = False
        for st in self.pool.states.values():
            if st.cfg.router:
                st.modes = self.modes[st.cfg.name] = self._make_modes(st)

    def _make_modes(self, st: BackendState) -> ModeManager:
        pol = st.cfg.mode_policy
        ctl = RouterCtl(lambda: self.session, st.cfg.url, lambda: _auth(st), pol.op_timeout_s, pol.poll_s,
                        ledger_dir=st.cfg.slotops_ledger_dir, engine_ids=st.cfg.model_engine_ids,
                        engine_owners=st.cfg.model_engine_owners, ledger_namespace=st.cfg.name)

        def health(ok: bool) -> None:
            if ok:
                self.pool.mark_ok(st)
            else:
                self.pool.mark_failure(st, hard=True)
        manager = ModeManager(st.cfg, ctl, self.metrics, health)
        # janela do handoff 35B: nada troca de modo no router enquanto ela esta aberta
        manager.block_switch = self.swap35.blocking_switch
        return manager

    def _probe_states(self) -> list[BackendState]:
        """O probe slot-telemetry nao fala com router (GET /slots sem model e invalido, com model autocarrega)."""
        return [st for st in self.pool.states.values() if not st.cfg.router]

    @web.middleware
    async def _request_metrics(self, request: web.Request, handler):
        if request.path != "/v1/chat/completions" or request.method != "POST":
            return await handler(request)
        route = "/v1/chat/completions"
        started = time.monotonic()
        status = "500"
        request_log.begin(request)
        try:
            response = await handler(request)
            status = str(response.status)
            return response
        except BaseException as exc:
            status = str(getattr(exc, "status", 500))
            raise
        finally:
            self.metrics._safe(self.metrics.http_request, route, status)
            self.metrics._safe(self.metrics.duration, route, time.monotonic() - started)
            request_log.finish(request, status)

    @web.middleware
    async def _client_auth(self, request: web.Request, handler):
        """Exige `Authorization: Bearer <chave>` se settings.client_key_env estiver definido (exceto /health)."""
        env = self.cfg.settings.client_key_env
        if env and request.path != "/health":
            want = os.environ.get(env, "")
            got = request.headers.get("Authorization", "")
            got = got[7:] if got.startswith("Bearer ") else ""
            if not want or not hmac.compare_digest(got.encode(), want.encode()):
                return web.json_response({"error": {"message": "unauthorized"}}, status=401)
        return await handler(request)

    def build(self) -> web.Application:
        app = web.Application(client_max_size=256 * 1024 * 1024, middlewares=[self._request_metrics, self._client_auth])
        app.router.add_post("/v1/chat/completions", self.chat)
        app.router.add_get("/v1/models", self.models)
        app.router.add_get("/health", self.health)
        app.router.add_get("/metrics", self.metrics_ep)
        app.on_startup.append(self._start)
        app.on_cleanup.append(self._stop)
        return app

    async def _start(self, _app) -> None:
        self._started = False
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=self.cfg.settings.connect_timeout_s),
            auto_decompress=False)
        try:
            await self.cold_handoff.reconcile_startup()
            self.sessions._purge_orphans()
            self._health_task = asyncio.create_task(self._health_loop())
            self.cold_handoff.start()
            if hasattr(self.probe, "attach"):
                self.probe.attach(self.session, _auth, self.metrics)
                await self.probe.detect(self._probe_states())   # capability slot-telemetry no start
            # Resolve each router-mode child ledger before marking App startup ready;
            # `sync()` itself stays fail-closed and the worker retries unresolved rows.
            for m in self.modes.values():
                await m.sync()
            for m in self.modes.values():
                m.start()
            # reconcilia a janela do 35B ANTES de ficar pronto (27B SEMPRE de pe)
            await self.swap35.reconcile()
            self.swap35.start()
            self._started = True
        except Exception:
            if self.session and not self.session.closed:
                await self.session.close()
            raise

    async def _stop(self, _app) -> None:
        self._started = False
        await self.swap35.stop()
        await self.cold_handoff.stop()
        for m in self.modes.values():
            await m.stop()
        if self._health_task:
            self._health_task.cancel()
            await asyncio.gather(self._health_task, return_exceptions=True)
            self._health_task = None
        if self.session:
            await self.session.close()

    # ---- saude -----------------------------------------------------------
    async def check_all(self) -> None:
        await asyncio.gather(*(self._check(st) for st in self.pool.states.values()))

    async def _check(self, st: BackendState) -> None:
        try:
            async with self.session.get(
                    st.cfg.url.rstrip("/") + "/health", headers=_auth(st),
                    timeout=aiohttp.ClientTimeout(total=self.cfg.settings.health_timeout_s)) as r:
                ok = r.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
            ok = False
        if ok and st.modes is not None and st.modes.broken:
            return      # router responde mas esta sem o modelo do modo: o ModeManager decide a volta
        if ok:
            if not st.healthy:
                log.info("backend %s voltou", st.cfg.name)
            self.pool.mark_ok(st)
        else:
            was = st.healthy
            self.pool.mark_failure(st)
            if was and not st.healthy:
                log.warning("backend %s removido (health falhou)", st.cfg.name)

    async def _health_loop(self) -> None:
        while True:
            try:
                await self.check_all()
            except Exception:
                log.exception("health loop")
            await asyncio.sleep(self.cfg.settings.health_interval_s)

    # ---- endpoints simples -----------------------------------------------
    async def health(self, _r):
        if not self._started:
            return web.json_response({"status": "down", "reason": "unready"}, status=503)
        up = [n for n, st in self.pool.states.items() if st.healthy]
        body = {"status": "ok" if up else "down", "backends_up": up,
                "backends_down": [n for n in self.pool.states if n not in up]}
        return web.json_response(body, status=200 if up else 503)

    async def models(self, _r):
        return web.json_response({"object": "list", "data": [
            {"id": m, "object": "model", "owned_by": "pool-proxy"} for m in self.cfg.models]})

    async def metrics_ep(self, _r):
        return web.Response(text=self.metrics.render(self.pool.states.values()),
                            content_type="text/plain", charset="utf-8")

    # ---- chat ------------------------------------------------------------
    async def chat(self, request: web.Request) -> web.StreamResponse:
        if not self._started:
            return web.json_response({"error": {"message": "Service unavailable: startup not complete"}}, status=503)
        raw = await request.read()
        try:
            body = json.loads(raw)
            if not isinstance(body, dict):
                raise ValueError
        except ValueError:
            return web.json_response({"error": {"message": "JSON invalido"}}, status=400)
        request_log.body(request, body, self.cfg.models)
        self.metrics._safe(self.metrics.request_n_observe, "/v1/chat/completions", body.get("n"))
        raw, body = _json_object_to_schema(raw, body)
        raw, body = self._normalize_schemas(request, raw, body)
        info = RequestInfo.from_body(body, self.cfg.settings)
        if request.headers.get(self.cfg.settings.low_priority_header, "").lower() == "low":
            return await self._chat_low(request, raw, body, info)
        await self._probe(info, body)
        if self.swap35.active:
            # pedido que nao e o 35B espera a janela na fila curta, sem forcar troca de modo
            await self.swap35.wait_free(info)
        handoff_resp = await self.cold_handoff.attempt(request, raw, body, info)
        if handoff_resp is None:
            handoff_resp = await self.swap35.attempt(request, raw, body, info)
        if handoff_resp is not None:
            return handoff_resp
        exclude: set[str] = set()
        spilled = False
        kv_orig = None
        last_err = "sem backend elegivel"
        for _ in range(self.cfg.settings.max_attempts + 1):
            dec = self.pool.route(info, exclude)
            if dec is None:
                break
            st, reason = dec
            self.metrics.decision(st.cfg.name, reason)
            self._note_affinity(info, st)
            lease = None
            if st.modes is not None:
                # demanda e espera pelo modo ANTES de ocupar um slot: com np pequeno, quem espera o
                # modelo segurando o slot impede os outros pedidos de registrarem demanda (sem troca)
                try:
                    lease = await st.modes.enter(st.modes.resolve(info.model))
                except ModelUnavailable as e:
                    last_err = str(e)
                    exclude.add(st.cfg.name)
                    self.metrics.request(st.cfg.name, "model_unavailable", 0.0)
                    continue
                finally:
                    # acquire() consumes the route reservation only after a successful mode wait.
                    if lease is None:
                        st.queued -= 1
            try:
                await self.pool.acquire(st)
            except asyncio.CancelledError:
                if lease is not None:
                    lease.release()
                raise
            except QueueTimeout:
                if lease is not None:
                    lease.release()
                    lease = None
                last_err = f"fila de {st.cfg.name} esgotada"
                self.metrics.request(st.cfg.name, "queue_timeout", self.cfg.settings.queue_timeout_s)
                if spilled or st.cfg.overflow:
                    break
                spilled = True
                dec = self.pool.spill_target(info, exclude)
                if dec is None:
                    break
                st, reason = dec
                self.metrics.decision(st.cfg.name, reason)
                try:
                    await self.pool.acquire(st)
                except QueueTimeout:
                    self.metrics.request(st.cfg.name, "queue_timeout", self.cfg.settings.queue_timeout_s)
                    break
            t0 = time.monotonic()
            status, ok = "error", False
            try:
                payload = self._with_slot(raw, body, info, st)
                if lease is not None:
                    payload = self._with_model(payload, body, st.modes.resolve(info.model))
                await self._flush_dirty(st, [k for k in list(st.slot_dirty) if k not in st.slot_busy])
                await self.handoff.before_forward(info, st)
                resp, status, cache_n = await self._forward(request, st, reason, payload)
                ok = True
                if status.startswith("2"):
                    self.pool.record_affinity(info, st.cfg.name)
                    self.metrics.affinity_result(st.cfg.name, info.reuse.get(st.cfg.name, 0), cache_n)
                if st.cfg.np == 1:
                    st.slot_sid = {0: self.sessions.sid(info)}
                else:
                    st.slot_sid.clear()   # nao sabemos qual slot o primeiro plano usou
                return resp
            except KvCapacity as e:
                ok = True
                status = "kv_capacity"
                if kv_orig is not None:
                    return kv_orig
                kv_orig = e.resp    # uma unica tentativa em outro backend; sem mark_failure
                exclude.add(st.cfg.name)
                continue
            except ModelUnavailable as e:
                last_err = str(e)
                exclude.add(st.cfg.name)
                status = "model_unavailable"
                continue
            except UpstreamDown as e:
                last_err = str(e)
                self.pool.mark_failure(st, hard=True)
                exclude.add(st.cfg.name)
                status = "connect_error"
                continue
            except asyncio.CancelledError:
                status = "client_cancel"
                raise
            finally:
                if lease is not None:
                    lease.release()
                dt = time.monotonic() - t0
                self.pool.release(st, dt if ok else None)
                self.metrics.request(st.cfg.name, status, dt)
        if kv_orig is not None:
            return kv_orig
        return web.json_response({"error": {"message": f"pool indisponivel: {last_err}"}}, status=503,
                                 headers={"Retry-After": "5"})

    async def _chat_low(self, request: web.Request, raw: bytes, body: dict, info: RequestInfo) -> web.StreamResponse:
        """Faixa de baixa prioridade (workers em lote): so ocupa vaos. Sem handoff, sem demanda de troca de modo,
        sem spill; espera (ate low_wait_s) em vez de falhar, inclusive com o backend fora do ar (janela de GPU)."""
        s = self.cfg.settings
        since = time.monotonic()
        deadline = since + s.low_wait_s
        low_ents: dict[str, list] = {}   # demanda low registrada por router (opt-in low_switch)
        await self._probe(info, body)    # prefixo reaproveitavel por backend (afinidade da faixa low)
        trace = {"since": since, "retries": 0}
        try:
            while True:
                try:
                    return await self._chat_low_loop(request, raw, body, info, deadline, low_ents, since, trace)
                except _LowRetry:
                    trace["retries"] += 1
                    await asyncio.sleep(s.low_poll_s)
        finally:
            for ent in low_ents.values():
                ent[2] = False
            if trace.get("backend"):
                log.info("decisao_low %s", json.dumps({
                    "backend": trace["backend"], "status": trace.get("status"), "est": info.est_prompt_tokens,
                    "exact": info.exact_tokens, "reuse": trace.get("reuse"), "owner": trace.get("owner"),
                    "cache_n": trace.get("cache_n"), "wait_s": round(trace.get("start", since) - since, 2),
                    "restored": trace.get("restored"), "restore_s": trace.get("restore_s"),
                    "handoff": trace.get("handoff"), "handoff_s": trace.get("handoff_s"),
                    "saved": trace.get("saved"), "save_s": trace.get("save_s"),
                    "svc_s": round(trace.get("svc", 0.0), 2), "retries": trace["retries"],
                    "msgs": len(info.messages)}, separators=(",", ":")))

    def _note_low_demand(self, info: RequestInfo, low_ents: dict) -> bool:
        """Registra (uma vez por router) demanda low pelo modelo fora do modo atual; True se algum router pode trocar."""
        possible = False
        for st in self.pool._eligible(info, set()):
            mgr = st.modes
            if mgr is None:
                continue
            m = mgr.resolve(info.model)
            if m is None or mgr.available(m) or not mgr.low_switch_possible(m):
                continue
            possible = True
            if st.cfg.name not in low_ents:
                low_ents[st.cfg.name] = mgr.record_low(m)
        return possible

    async def _chat_low_loop(self, request, raw, body, info, deadline, low_ents, since=None,
                             trace=None) -> web.StreamResponse:
        s = self.cfg.settings
        trace = trace if trace is not None else {}
        while True:
            now = time.monotonic()
            kv_exclude = trace.get("kv_exclude", frozenset())
            if kv_exclude and not self.pool.low_eligible(info, kv_exclude):
                return trace["kv_orig"]     # 503 kv_capacity_error sem outro backend elegivel
            if kv_exclude:
                st = self.pool.low_candidate(info, now, since, kv_exclude)
            else:
                st = self.pool.low_candidate(info, now, since)
            lease = None
            if st is not None and st.modes is not None:
                lease = st.modes.try_enter_low(st.modes.resolve(info.model))
                if lease is None:
                    st = None
            if st is not None:
                break
            switchable = self._note_low_demand(info, low_ents)
            if not switchable and self._low_model_parked(info):
                self.metrics.request("low", "low_wrong_mode", 0.0)
                return web.json_response(
                    {"error": {"message": "pool: modelo fora do modo atual do router (pedido low nao provoca troca); "
                                          "use o modelo carregado"}}, status=409)
            if now >= deadline:
                self.metrics.request("low", "low_timeout", s.low_wait_s)
                if trace.get("kv_orig") is not None:
                    return trace["kv_orig"]
                return web.json_response({"error": {"message": "pool: sem vao para pedido de baixa prioridade"}},
                                         status=503, headers={"Retry-After": "30"})
            await asyncio.sleep(s.low_poll_s)
        st.queued += 1
        st.low_inflight += 1             # reserva a vaga low antes de esperar o slot (evita dois low no mesmo vao)
        try:
            await self.pool.acquire(st)
        except BaseException:
            st.low_inflight -= 1
            raise
        self.metrics.decision(st.cfg.name, "low")
        self._note_affinity(info, st)
        eff = self.pool.low_reuse(info)
        t0 = time.monotonic()
        trace.update(backend=st.cfg.name, start=t0, reuse=eff.get(st.cfg.name, 0),
                     owner=max(eff, key=eff.get) if eff and max(eff.values()) > 0 else None)
        status, ok = "error", False
        slot = None
        detached = None
        try:
            payload = self._with_slot(raw, body, info, st)
            slot = await self._low_session_slot(st, info, trace, body)
            if slot is not None:
                payload = json.dumps({**json.loads(payload), "id_slot": slot, "cache_prompt": True},
                                     ensure_ascii=False).encode()
            if lease is not None:
                payload = self._with_model(payload, body, st.modes.resolve(info.model))
            resp, status, cache_n = await self._forward(request, st, "low", payload, retry_unavailable=True)
            trace["cache_n"] = cache_n
            ok = True
            if slot is not None:
                st.slot_sid[slot] = self.sessions.sid(info)
                st.slot_tokens[slot] = self._need_tokens(info)
                st.slot_seq += 1
                st.slot_used[slot] = st.slot_seq
            elif st.cfg.np == 1:
                st.slot_sid = {0: self.sessions.sid(info)}
            if str(status).startswith("2"):
                self.pool.record_affinity(info, st.cfg.name)
                # o cliente costuma fechar a conexao logo depois do fim do stream e handler_cancellation cancelaria o
                # save (2026-10-11: ~1/3 dos turnos low como low_client_cancel e quase todo save=0); o save segue
                # blindado e o slot/vaga so sao liberados quando ele termina
                save = asyncio.ensure_future(self._low_session_save(st, info, slot, trace))
                try:
                    await asyncio.shield(save)
                except asyncio.CancelledError:
                    detached = save
                    raise
            return resp
        except KvCapacity as e:
            status = "low_kv_capacity"
            if trace.get("kv_orig") is not None:
                return trace["kv_orig"]
            trace["kv_orig"] = e.resp       # uma unica tentativa em outro backend; sem mark_failure
            trace["kv_exclude"] = frozenset({st.cfg.name})
            if time.monotonic() < deadline:
                raise _LowRetry() from None
            return e.resp
        except (ModelUnavailable, UpstreamDown) as e:
            if time.monotonic() < deadline:
                # erro antes de responder (conexao, 502/503, modelo indisponivel): volta a esperar um vao; low nao falha
                # por janela de GPU nem por recarga do modelo enquanto houver prazo
                status = "low_retry"
                raise _LowRetry() from None
            status = "low_error"
            if trace.get("kv_orig") is not None:
                return trace["kv_orig"]
            return web.json_response({"error": {"message": f"pool: {e}"}}, status=503, headers={"Retry-After": "30"})
        except asyncio.CancelledError:
            if detached is None:
                status = "low_client_cancel"
            raise
        finally:
            if detached is not None:
                detached.add_done_callback(
                    lambda _f: self._low_release(st, slot, lease, t0, status, ok, trace))
            else:
                self._low_release(st, slot, lease, t0, status, ok, trace)

    def _low_release(self, st, slot, lease, t0, status, ok, trace) -> None:
        if slot is not None:
            st.slot_busy.discard(slot)
        if lease is not None:
            lease.release()
        dt = time.monotonic() - t0
        trace.update(status=status, svc=dt)
        self.pool.release(st, dt if ok else None, low=True)
        self.metrics.request(st.cfg.name, status, dt)

    async def _low_session_slot(self, st: BackendState, info: RequestInfo, trace: dict, body: dict) -> int | None:
        """Slot fixado para o pedido low (sessoes ligadas): o slot onde a sessao esta residente; senao um slot livre
        (vazio ou o menos recente), restaurando nele o estado salvo da sessao se houver. Falha no restore => prefill."""
        sess = self.sessions
        if not sess.enabled():
            return None
        sid = sess.sid(info)
        if st.modes is not None and st.modes.epoch != st.slot_epoch:
            st.slot_dirty.clear()
            st.slot_tokens.clear()
            st.slot_sid.clear()              # modelo do router caiu/reiniciou desde o registro (2026-10-11: restart do
            st.slot_epoch = st.modes.epoch   # router deixou slot_sid dizendo "residente" com o slot vazio => 109 s)
        free = [k for k in range(st.cfg.np) if k not in st.slot_busy]
        if sid is None or not free:
            return None
        # o probe le /slots do servidor: e a verdade sobre onde o prefixo esta. O rastreamento interno (slot_sid) se
        # perde com pedidos de primeiro plano e saves desligados; 2026-10-11 04:03 ele mandou a sessao para o slot 0
        # com o prefixo (37k) no slot 1 => prefill completo de 298 s nas MI50
        seen = info.slot.get(st.cfg.name)
        if (seen is not None and seen in free and info.reuse.get(st.cfg.name, 0) >= self.cfg.settings.low_session_min_tokens):
            st.slot_busy.add(seen)
            st.slot_sid[seen] = sid
            await self._fit_unified(st, seen, info)
            return seen
        mine = [k for k in free if st.slot_sid.get(k) == sid]
        if mine and st.cfg.name in info.reuse:
            mine = []                        # o probe viu os slots e nao achou o prefixo (acima): o registro esta velho
        if mine:
            st.slot_busy.add(mine[0])
            await self._fit_unified(st, mine[0], info)
            return mine[0]
        slot = min(free, key=lambda k: (k in st.slot_sid, st.slot_used.get(k, 0)))
        st.slot_busy.add(slot)
        await self._flush_dirty(st, [slot])  # outra sessao vai sobrescrever: salva a residente se o arquivo esta atras
        st.slot_sid.pop(slot, None)
        st.slot_tokens.pop(slot, None)
        await self._fit_unified(st, slot, info)          # o conteudo antigo deixa de valer (sera sobrescrito)
        e = sess.pin_lookup(info)
        if e is None:
            await self._low_handoff(st, info, body, slot, None, trace)
            return slot
        t = time.monotonic()
        try:
            try:
                ok = await sess.restore_into(st, e, slot)
            except Exception as ex:   # noqa: BLE001 - qualquer falha cai no prefill
                log.info("sessao: restore falhou em %s (%s)", st.cfg.name, type(ex).__name__)
                ok = False
            trace.update(restored=e["tokens"] if ok else 0, restore_s=round(time.monotonic() - t, 2))
            # Hold the pin through any source restore that consumes this snapshot.
            await self._low_handoff(st, info, body, slot, e if ok else None, trace)
            return slot
        finally:
            sess.unpin(e)

    async def _low_handoff(self, st: BackendState, info: RequestInfo, body: dict, slot: int, e: dict | None,
                           trace: dict) -> bool:
        """Faixa low no destino do handoff (MI50) sem o prefixo residente: se faltam >= low_handoff_min_tokens de
        prefill e a origem (3090) esta ociosa, a origem restaura a sessao (se houver), prefilla so o que falta, salva,
        e o estado e restaurado no slot fixado do destino; o decode fica no destino. A 3090 so fica presa no prefill.
        False => caminho normal (restore da sessao no destino ou prefill no destino)."""
        s, ch, pool = self.cfg.settings, self.cold_handoff, self.pool
        src = pool.states.get(s.handoff_source)
        if (not s.low_handoff or not s.handoff_enabled or st.cfg.name != s.handoff_target or src is None
                or info.has_image or "id_slot" in body):
            return False
        model = ch.model_of(st)
        if model and st.modes.resolve(info.model) != model:
            return False
        have = e["tokens"] if e and e.get("fmt") == st.cfg.handoff_state_format else 0
        est = info.est_prompt_tokens * (1.0 if info.exact_tokens else s.low_handoff_est_factor)
        if est - have < s.low_handoff_min_tokens:
            return False
        sid = self.sessions.sid(info)

        def free_slots():
            return [k for k in range(src.cfg.np) if k not in src.slot_busy]
        # basta UM slot livre na origem e nenhum pedido de primeiro plano nela (com np 2 a 3090 quase nunca fica toda
        # ociosa; exigir isso fazia o pedido esperar low_handoff_wait_s e prefillar frio nas MI50 do mesmo jeito)
        until = time.monotonic() + s.low_handoff_wait_s
        while (src not in pool._eligible(info, set()) or src.queued or src.handoff_busy
               or src.inflight - src.low_inflight > 0 or src.inflight >= src.cfg.np or not free_slots()):
            if time.monotonic() >= until:
                self.metrics.handoffs["low_source_busy"] += 1
                return False
            await asyncio.sleep(0.2)
        fs = free_slots()
        mine = [k for k in fs if src.slot_sid.get(k) == sid]
        src_slot = mine[0] if mine else min(fs, key=lambda k: (k in src.slot_sid, src.slot_used.get(k, 0)))
        src.slot_busy.add(src_slot)        # sem await desde a checagem: reserva atomica do slot da origem
        await src.sem.acquire()            # inflight < np: ha permissao livre, nao suspende
        src.inflight += 1
        src.low_inflight += 1
        held = 1
        filename = "pp-handoff-" + uuid.uuid4().hex + ".bin"
        stages: dict = {}
        progress: dict = {}
        t = time.monotonic()
        try:
            if e is not None and src.slot_sid.get(src_slot) != sid:
                await self.sessions.restore_into(src, e, src_slot)   # falhou => a origem prefilla tudo
            if src.slot_sid.get(src_slot) != sid:
                await self._flush_dirty(src, [src_slot])
            src.slot_dirty.pop(src_slot, None)
            src.slot_sid.pop(src_slot, None)
            await self._fit_unified(src, src_slot, info)
            # sem timeout externo: prepare() limita o prefill pelo orcamento derivado dos tokens exatos
            _slot, count = await ch.prepare(src, st, body, filename, stages=stages, dst_slot=slot,
                                            src_slot=src_slot, progress=progress)
            src.slot_sid[src_slot] = sid
            src.slot_tokens[src_slot] = self._need_tokens(info)
            src.slot_seq += 1
            src.slot_used[src_slot] = src.slot_seq
            self.metrics.handoffs["low_ok"] += 1
            self.metrics.handoff_stages(stages)
            trace.update(handoff=count - 1, handoff_s=round(time.monotonic() - t, 2))
            log.info("handoff low: %d tokens, etapas %s", count - 1,
                     " ".join(f"{k}={v:.2f}s" for k, v in stages.items()))
            return True
        except Exception as ex:   # noqa: BLE001 - qualquer falha cai no caminho normal
            log.info("handoff low: falhou (%s: %s); caminho normal", type(ex).__name__, ex)
            if progress.get("src_slot") is not None and "prefill_save" in stages:
                await ch.quiesce(src, progress.get("src_slot"))   # deixa o prefixo parcial reaproveitavel (LCP)
            self.metrics.handoffs["low_fallback"] += 1
            trace.update(handoff=0, handoff_s=round(time.monotonic() - t, 2))
            return False
        finally:
            for _ in range(held):
                pool.release(src, None, low=True)
            src.slot_busy.discard(src_slot)
            ch.cleanup(filename)

    def _need_tokens(self, info: RequestInfo) -> int:
        f = 1.0 if info.exact_tokens else self.cfg.settings.low_handoff_est_factor
        return int(info.est_prompt_tokens * f) + 4096          # folga para a saida e o raciocinio

    async def _fit_unified(self, st: BackendState, slot: int, info: RequestInfo) -> None:
        """KV unificado (3090 -np 2, MI50 kv-unified): duas sessoes longas nao cabem juntas. Antes do turno, se a sessao
        somada as residentes nos outros slots passa de kv_unified_ctx, salva (se suja) e apaga as outras, das mais
        antigas para as mais novas, ate caber. Slots em uso por outro pedido ficam."""
        cap = st.cfg.kv_unified_ctx
        if cap <= 0 or st.cfg.np < 2:
            return
        need = self._need_tokens(info)
        others = sorted((k for k in st.slot_tokens if k != slot and k not in st.slot_busy),
                        key=lambda k: st.slot_used.get(k, 0))
        used = sum(v for k, v in st.slot_tokens.items() if k != slot)
        for k in others:
            if need + used <= cap:
                break
            await self._flush_dirty(st, [k])
            try:
                await self.cold_handoff.req(st, "POST", f"/slots/{k}?action=erase", {})
            except Exception as ex:   # noqa: BLE001
                log.info("sessao: erase do slot %s em %s falhou (%s)", k, st.cfg.name, type(ex).__name__)
                raise RuntimeError("unified KV erase did not reach a verified terminal result") from ex
            used -= st.slot_tokens.pop(k, 0)
            st.slot_sid.pop(k, None)
            log.info("sessao: slot %s de %s liberado para caber %d tokens (KV unificado %d)", k, st.cfg.name, need, cap)

    async def _flush_dirty(self, st: BackendState, slots) -> None:
        """Salva os slots com turno adiado (low_session_save_every_tokens) antes que sejam sobrescritos. O chamador
        detem o slot (ou todos os slots do backend)."""
        for k in slots:
            info = st.slot_dirty.pop(k, None)
            if info is None:
                continue
            t = time.monotonic()
            try:
                e = await self.sessions.save(st, info, k)
            except Exception as ex:   # noqa: BLE001
                log.info("sessao: save adiado falhou em %s (%s)", st.cfg.name, type(ex).__name__)
                e = None
            log.info("sessao: save adiado em %s slot %s: %s tokens em %.2fs", st.cfg.name, k, e["tokens"] if e else 0,
                     time.monotonic() - t)

    async def _low_session_save(self, st: BackendState, info: RequestInfo, slot: int | None, trace: dict) -> None:
        """Depois do turno, ainda com o slot reservado: salva o KV da sessao (so backends de low_session_save_from)."""
        sess = self.sessions
        if not sess.enabled() or st.cfg.name not in self.cfg.settings.low_session_save_from:
            return
        if slot is None:
            return                # sem slot fixado nao se sabe qual slot atendeu
        every = self.cfg.settings.low_session_save_every_tokens
        if every > 0:
            prev = sess.entries.get(sess.sid(info) or "")
            now_tok = trace.get("cache_n") or 0
            if prev and prev.get("origin_slot") == (st.cfg.name, slot) and sess.lookup(info) is prev                     and now_tok - prev["tokens"] < every:
                st.slot_dirty[slot] = info     # arquivo ainda proximo: adia; salvo antes de o slot ser sobrescrito
                trace.update(saved=-1)
                return
        st.slot_dirty.pop(slot, None)
        t = time.monotonic()
        try:
            e = await sess.save(st, info, slot)
        except Exception as ex:   # noqa: BLE001
            log.info("sessao: save falhou em %s (%s)", st.cfg.name, type(ex).__name__)
            e = None
        trace.update(saved=e["tokens"] if e else 0, save_s=round(time.monotonic() - t, 2))
        log.info("sessao: save em %s slot %s: %s tokens em %.2fs", st.cfg.name, slot, trace["saved"], trace["save_s"])

    def _low_model_parked(self, info: RequestInfo) -> bool:
        """O modelo so existe em routers saudaveis cujo modo atual nao o serve (e nada esta trocando): esperar seria
        indefinido, porque low nunca provoca troca."""
        hosts = [st for st in self.pool._eligible(info, set()) if st.modes is not None or not st.cfg.router]
        if not hosts or any(st.modes is None for st in hosts):
            return False
        return all(st.modes.switching is None and not st.modes.broken and st.modes.mode is not None
                   and not st.modes.available(st.modes.resolve(info.model)) for st in hosts)

    def _normalize_schemas(self, request: web.Request, raw: bytes, body: dict):
        """ancora `pattern` (e repara o que o conversor do llama.cpp recusa) ANTES de
        qualquer uso do corpo — decode, /apply-template do handoff, probe e request-log.
        Corpo que nao precisa de reparo sai byte a byte identico."""
        if not self.cfg.settings.schema_normalize:
            return raw, body
        raw, body, stats = schemas.normalize_body(raw, body,
                                                  cap=self.cfg.settings.schema_minmax_cap)
        if stats.events:
            for action, reason, n in stats.events:
                self.metrics._safe(self.metrics.schema_event, action, reason, n)
            # so contagens: a politica quebrada e' do cliente e o conteudo nunca vai ao log
            log.info("schema: corpo reparado antes do forward (%s)", stats.summary())
            request_log.schema(request, stats.events)
        return raw, body

    async def _probe(self, info: RequestInfo, body: dict) -> None:
        """Preenche info com tokens exatos e prefixo reaproveitavel por backend. Nunca falha o pedido."""
        if not hasattr(self.probe, "analyze"):    # SlotProbe legado: so find_owner
            try:
                info.hint = await self.probe.find_owner(self._probe_states(), info.messages)
            except Exception:
                log.exception("probe")
            return
        try:
            pr: ProbeResult = await self.probe.analyze(self._probe_states(), body)
        except Exception:
            log.exception("probe")
            return
        if pr.prompt_tokens is not None:
            info.est_prompt_tokens, info.exact_tokens = pr.prompt_tokens, True
        for name, a in pr.backends.items():
            info.reuse[name], info.slot[name] = a.reuse, a.slot

    def _note_affinity(self, info: RequestInfo, st: BackendState) -> None:
        if not info.reuse:
            return
        best = max(info.reuse.values())
        mine = info.reuse.get(st.cfg.name, 0)
        outcome = "cold" if best <= 0 else ("owner" if mine >= best else "skipped")
        self.metrics.affinity_route(outcome, st.cfg.name, mine)

    @staticmethod
    def _with_slot(raw: bytes, body: dict, info: RequestInfo, st: BackendState) -> bytes:
        """Injeta id_slot do slot de maior afinidade (so backend slot-telemetry, reuse>0, cliente nao fixou)."""
        slot = info.slot.get(st.cfg.name)
        if slot is None or info.reuse.get(st.cfg.name, 0) <= 0 or "id_slot" in body:
            return raw
        return json.dumps({**body, "id_slot": slot}, ensure_ascii=False).encode()

    @staticmethod
    def _with_model(raw: bytes, body: dict, model: str) -> bytes:
        """Backend router: o corpo leva o nome canonico do modelo (alias/default resolvidos)."""
        if body.get("model") == model:
            return raw
        return json.dumps({**json.loads(raw), "model": model}, ensure_ascii=False).encode()

    async def _forward(self, request: web.Request, st: BackendState, reason: str, raw: bytes,
                       handoff_cache: int | None = None, retry_unavailable: bool = False):
        hdrs = {k: v for k, v in request.headers.items() if k.lower() not in DROP_REQ}
        hdrs.update(_auth(st))
        hdrs["Content-Type"] = "application/json"
        url = st.cfg.url.rstrip("/") + "/v1/chat/completions"
        try:
            up = await self.session.post(url, data=raw, headers=hdrs)
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            raise UpstreamDown(f"{st.cfg.name}: {type(e).__name__}") from None
        request_log.backend(request, st.cfg.name)
        out_h = {k: v for k, v in up.headers.items() if k.lower() not in DROP_RESP}
        out_h["X-Pool-Backend"] = st.cfg.name
        out_h["X-Pool-Reason"] = reason
        try:
            if up.status == 503 and "text/event-stream" not in up.headers.get("Content-Type", ""):
                data = await up.read()
                if handoff_cache is None and b"kv_capacity_error" in data:
                    self.metrics._safe(self.metrics.kv_refusal, "/v1/chat/completions")
                    raise KvCapacity(web.Response(status=503, body=data, headers=out_h))
            if retry_unavailable and up.status in (502, 503):
                # faixa low: engine recarregando ("Loading model") ou saindo de janela de GPU; o pedido espera de novo
                raise UpstreamDown(f"{st.cfg.name}: HTTP {up.status}")
            if handoff_cache is not None and up.status >= 400:
                from .handoff import HandoffError
                raise HandoffError("decode HTTP error")
            if "text/event-stream" in up.headers.get("Content-Type", ""):
                resp = web.StreamResponse(status=up.status, headers=out_h)
                await resp.prepare(request)
                tail = b""
                reqlog_sse = request_log.SSE(request)
                try:
                    async for chunk in up.content.iter_any():
                        reqlog_sse.feed(chunk)
                        tail = (tail + chunk)[-8192:]
                        await resp.write(chunk)
                    await resp.write_eof()
                except (ConnectionResetError, aiohttp.ClientConnectionError):
                    pass  # cliente saiu; o finally fecha a conexao a montante
                return resp, str(up.status), _cache_n(tail)
            try:
                data = await up.read()
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
                if handoff_cache is not None:
                    from .handoff import HandoffError
                    raise HandoffError("decode body unavailable") from None
                raise
            request_log.response(request, data)
            if handoff_cache is not None:
                observed = _cache_n(data[-8192:])
                if observed is not None and observed < handoff_cache:
                    from .handoff import HandoffError
                    self.metrics.handoffs["cache_miss"] += 1
                    raise HandoffError("decode cache miss")
            return (web.Response(status=up.status, body=data, headers=out_h), str(up.status),
                    _cache_n(data[-8192:]))
        finally:
            up.close()  # desconexao/cancelamento: derruba o socket a montante (engine cancela a geracao)


class _LowRetry(Exception):
    """Pedido low que falhou antes de responder e ainda tem prazo: refaz a selecao."""


def run_app_kwargs() -> dict:
    # handler_cancellation: desconexao do cliente cancela o handler mesmo sem escrita (prefill longo)
    return {"handler_cancellation": True}


def main() -> None:
    ap = argparse.ArgumentParser(description="pool proxy v1")
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = cfgmod.load(a.config)
    web.run_app(App(cfg).build(), host=a.host or cfg.host, port=a.port or cfg.port, **run_app_kwargs())


if __name__ == "__main__":
    main()
