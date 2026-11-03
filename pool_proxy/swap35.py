"""Handoff do Qwen3.6-35B-A3B: prefill na RTX 3090 (engine de troca) -> decode nas MI50.

O 35B (22,3 GB de pesos+KV na 3090) NAO coexiste com o 27B residente (18,5 GB). Por isso o
caminho tem uma janela: o `prefill35-ctl` para o `swap engine` COM DRENO, sobe o engine
35B de prefill, e o pool-proxy faz prefill+save, restaura o slot no `mi50-35b` do router e
decodifica nas MI50. Em QUALQUER falha o pedido segue para o 35B direto nas MI50 e o 27B volta
para a 3090 (garantia do ctl: finally + TTL + ledger/reconcile no boot e no startup do proxy).

Gate (todos obrigatorios, checados antes de reservar qualquer coisa):
  * modelo do pedido em `handoff_35b.models` (so o 35B);
  * prompt estimado >= `min_tokens` (64000);
  * 3090 ociosa: sem inflight, sem fila e sem outro handoff em curso;
  * sessao nova / sem cache util no destino (reuso previsto 0 e nenhum slot do MI50 com
    >= `max_cached_tokens` do prefixo);
  * router consegue servir `router_model` (modo m35) pela politica do ModeManager;
  * uma troca por vez (flag `active`, sem await entre checar e reservar).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path

import aiohttp

from .handoff import HandoffError
from .routerctl import RouterError
from .slotops import SlotOperationError, SlotOperations

log = logging.getLogger("pool_proxy.swap35")


class CtlError(Exception):
    def __init__(self, message: str, status: int | None = None, body=None):
        super().__init__(message)
        self.status = status
        self.body = body


class Handoff35Error(Exception):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


class CtlClient:
    """Cliente HTTP do prefill35-ctl + passthrough para o engine de prefill."""

    def __init__(self, app, settings):
        self.app = app
        self.s = settings

    def _headers(self):
        key = os.environ.get(self.s.ctl_key_env, "") if self.s.ctl_key_env else ""
        return {"Authorization": f"Bearer {key}"} if key else {}

    async def call(self, method, path, body=None, timeout=None):
        if self.app.session is None:
            raise CtlError("no session")
        url = self.s.ctl_url.rstrip("/") + path
        try:
            async with self.app.session.request(
                    method, url, json=body, headers=self._headers(),
                    timeout=aiohttp.ClientTimeout(total=timeout or self.s.ctl_timeout_s)) as r:
                raw = await r.read()
                try:
                    data = json.loads(raw or b"{}")
                except ValueError:
                    data = {"raw": raw[:300].decode("utf-8", "replace")}
                if r.status < 200 or r.status >= 300:
                    raise CtlError(f"HTTP {r.status}", status=r.status, body=data)
                return data
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            raise CtlError(f"{type(exc).__name__}", status=None) from None

    async def engine(self, method, path, body=None, timeout=None):
        """Chamada ao engine 35B atraves do ctl (loopback no 221, so com a janela aberta)."""
        return await self.call(method, "/engine" + path, body, timeout=timeout)


class Swap35:
    def __init__(self, app):
        self.app = app
        self.s = app.cfg.settings.handoff_35b
        self.ctl = CtlClient(app, self.s) if self.s.enabled else None
        self._free = asyncio.Event()
        self._free.set()
        self.active = False                # janela aberta (35B na 3090)
        self.ready = False                 # provado por /health do ctl (startup + health loop)
        self.stages: dict[str, float] = {}
        self.last_error = ""
        self.holds = 0
        self.hold_wait_s = 0.0
        self._hold_timeouts = 0
        self._task: asyncio.Task | None = None
        self._source_ops: SlotOperations | None = None

    # ---- consulta ----------------------------------------------------------
    def enabled(self) -> bool:
        return bool(self.s.enabled and self.ctl is not None)

    def is_27b(self, info) -> bool:
        """Pedido que a janela do 35B tira do ar (tudo que nao e o modelo do handoff)."""
        return info.model not in self.s.models

    def blocking_switch(self) -> bool:
        """ModeManager consulta: nao trocar de modo do router enquanto a janela esta aberta."""
        return self.active

    def _metric(self, name: str, n: int = 1) -> None:
        try:
            self.app.metrics.handoffs[name] += n
        except Exception:
            pass

    def _stages(self, stages) -> None:
        try:
            self.app.metrics.handoff_stages(stages)
        except Exception:
            pass

    # ---- reconciliacao / saude --------------------------------------------
    async def reconcile(self) -> None:
        """No startup do proxy, ANTES de ficar pronto: fecha janela orfa e prova o 27B."""
        if not self.enabled():
            return
        try:
            state = await self.ctl.call("GET", "/state", timeout=self.s.reconcile_timeout_s)
        except CtlError as exc:
            self.ready = False
            self.last_error = f"startup: {exc}"
            log.warning("handoff35: ctl indisponivel no startup (%s); gate desligado", exc)
            self._metric("ctl_unreachable")
            return
        if state.get("open"):
            log.warning("handoff35: janela aberta de sessao anterior (request_id=%s age=%ss); fechando",
                        state.get("request_id"), state.get("age_s"))
            try:
                res = await self.ctl.call("POST", "/close",
                                          {"token": state.get("token"), "reason": "proxy_startup"},
                                          timeout=self.s.close_timeout_s)
            except CtlError as exc:
                self.last_error = f"reconcile: {exc}"
                log.error("handoff35: falha ao fechar janela orfa (%s)", exc)
                self._metric("reconcile_failed")
                self.ready = False
                return
            self._metric("reconciled" if res.get("ok") else "reconcile_unrestored")
        await self._refresh_ready()

    async def _refresh_ready(self) -> None:
        try:
            h = await self.ctl.call("GET", "/health", timeout=self.s.reconcile_timeout_s)
        except CtlError as exc:
            self.ready = False
            self.last_error = f"health: {exc}"
            return
        ok = bool(h.get("ok"))
        if ok and not self.ready:
            log.info("handoff35: ctl pronto (%s)", h)
        if not ok:
            self.last_error = f"ctl nao pronto: {h}"
        self.ready = ok

    def start(self) -> None:
        if self.enabled():
            self._task = asyncio.create_task(self._health_loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    async def _health_loop(self) -> None:
        while True:
            await asyncio.sleep(self.s.health_interval_s)
            try:
                state = await self.ctl.call("GET", "/state", timeout=self.s.reconcile_timeout_s)
                if state.get("open") and not self.active:
                    log.warning("handoff35: janela aberta sem dono neste proxy; fechando")
                    await self.ctl.call("POST", "/close",
                                        {"token": state.get("token"), "reason": "orphan"},
                                        timeout=self.s.close_timeout_s)
                    self._metric("orphan_closed")
                await self._refresh_ready()
            except CtlError as exc:
                self.ready = False
                self.last_error = f"health: {exc}"
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("handoff35: health loop")

    # ---- espera dos pedidos 27B durante a janela ---------------------------
    async def wait_free(self, info, timeout=None) -> float:
        """Pedido que nao e do 35B espera a janela fechar (fila curta), sem forcar troca de modo."""
        if not self.active or not self.is_27b(info):
            return 0.0
        budget = self.s.hold_wait_s if timeout is None else timeout
        if budget <= 0:
            return 0.0
        t = time.monotonic()
        self.holds += 1
        result = "free"
        try:
            await asyncio.wait_for(self._free.wait(), budget)
        except asyncio.TimeoutError:
            self._hold_timeouts += 1
            result = "timeout"
            self._metric("hold_timeout")
        dt = time.monotonic() - t
        self.hold_wait_s += dt
        self._metric("hold_" + result)
        return dt

    # ---- gate de cache do destino -----------------------------------------
    async def _target_cached(self, mgr) -> int | None:
        """Maior prefixo residente num slot do modelo do handoff no router (None = nao provado)."""
        model = self.s.router_model
        if not mgr.available(model):
            return 0                      # modelo fora do modo atual: nao ha cache do 35B no MI50
        try:
            slots = await mgr.ctl.slots(model)
        except (RouterError, aiohttp.ClientError, asyncio.TimeoutError, OSError):
            return None
        best = 0
        for s in slots:
            if not isinstance(s, dict) or s.get("is_processing"):
                continue
            n = s.get("n_cached_tokens", s.get("n_prompt_tokens", 0))
            if type(n) is int and n > best:
                best = n
        return best

    async def _target_contract(self, mgr) -> None:
        """O preset carregado no destino tem de declarar o mesmo contrato de KV/np do prefill."""
        model = self.s.router_model
        try:
            data = await mgr.ctl._req("GET", "/models", params={"autoload": "false"})
        except (RouterError, aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            raise Handoff35Error("target_models", str(exc)) from None
        entry = next((m for m in ((data or {}).get("data") or []) if m.get("id") == model), None)
        status = (entry or {}).get("status") or {}
        if status.get("value") != "loaded":
            raise Handoff35Error("target_not_loaded", str(status.get("value")))
        preset = {}
        for line in str(status.get("preset", "")).splitlines():
            k, sep, v = line.partition("=")
            if sep:
                preset[k.strip()] = v.strip().lower()
        for k, v in self.s.expected_preset.items():
            if preset.get(k) != str(v).lower():
                raise Handoff35Error("target_preset", f"{k}={preset.get(k)!r}")
        if preset.get("kv-unified") in ("true", "1", "on"):
            raise Handoff35Error("target_preset", "kv-unified ligado (o arquivo de slot carrega n_stream)")
        par = preset.get("parallel")
        if par is not None and int(par) != self.s.target_np:
            raise Handoff35Error("target_preset", f"parallel={par}")

    # ---- tokens / slots ---------------------------------------------------
    async def _tokens(self, body, source: bool) -> list[int]:
        send = {k: v for k, v in body.items() if k != "model"}
        try:
            if source:
                templated = await self.ctl.engine("POST", "/apply-template", send,
                                                  timeout=self.s.tokens_timeout_s)
                result = await self.ctl.engine("POST", "/tokenize",
                                               {"content": templated.get("prompt", ""),
                                                "add_special": True, "parse_special": True},
                                               timeout=self.s.tokens_timeout_s)
            else:
                req = self.app.modes[self.s.target].ctl.slotop_request(self.s.router_model)
                templated = await req("POST", "/apply-template", send)
                result = await req("POST", "/tokenize",
                                   {"content": templated.get("prompt", ""),
                                    "add_special": True, "parse_special": True})
        except (CtlError, RouterError, aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            raise Handoff35Error("tokenize", str(exc)) from None
        tokens = result.get("tokens") if isinstance(result, dict) else None
        if not isinstance(tokens, list) or not tokens or any(type(t) is not int for t in tokens):
            raise Handoff35Error("tokenize", "tokens invalidos")
        return tokens

    async def _source_slot(self) -> int:
        try:
            slots = await self.ctl.engine("GET", "/slots", timeout=self.s.stage_timeout_s)
        except CtlError as exc:
            raise Handoff35Error("source_slots", str(exc)) from None
        if not isinstance(slots, list) or len(slots) != self.s.source_np:
            raise Handoff35Error("source_slots", f"capacidade {len(slots) if isinstance(slots, list) else '?'}")
        idle = [s for s in slots if isinstance(s, dict) and s.get("is_processing") is False
                and type(s.get("id")) is int]
        if not idle:
            raise Handoff35Error("source_slots", "sem slot ocioso")
        return min(idle, key=lambda s: (s.get("n_cached_tokens", 0), s["id"]))["id"]

    def _source_operations(self) -> SlotOperations:
        if self._source_ops is None:
            db = Path(self.s.ledger_dir) / "handoff35-prefill-slotops.sqlite3"
            self._source_ops = SlotOperations(
                db, "handoff35:prefill", timeout_s=self.s.stage_timeout_s,
                engine_id=self.s.engine_id, engine_owner=self.s.engine_owner,
                endpoint=self.s.ctl_url, model="")
        return self._source_ops

    async def _engine_request(self, method, path, body):
        try:
            return await self.ctl.engine(method, path, body, timeout=self.s.stage_timeout_s)
        except CtlError as exc:
            raise SlotOperationError(f"prefill engine: {exc}", status=exc.status) from None

    async def _dst_slot(self, mgr) -> int:
        try:
            slots = await mgr.ctl.slots(self.s.router_model)
        except (RouterError, aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            raise Handoff35Error("target_slots", str(exc)) from None
        st = self.app.pool.states[self.s.target]
        busy = set(getattr(st, "slot_busy", ()))
        # `has_media` nao existe em toda versao do /slots: ausente == sem midia
        idle = [s for s in slots if isinstance(s, dict) and s.get("is_processing") is False
                and s.get("has_media") is not True and type(s.get("id")) is int
                and s["id"] not in busy]
        if not idle:
            raise Handoff35Error("target_slots", "sem slot ocioso")
        return min(idle, key=lambda s: (s.get("n_cached_tokens", 0), s["id"]))["id"]

    # ---- caminho principal ------------------------------------------------
    async def attempt(self, request, raw, body, info):
        s, pool = self.s, self.app.pool
        if not self.enabled() or not self.ready:
            return None
        if info.model not in s.models:
            return None
        if info.has_image or info.hint or "id_slot" in body:
            self._metric("skip_shape")
            return None
        if info.est_prompt_tokens < s.min_tokens:
            return None
        if max(info.reuse.values(), default=0) > 0:
            self._metric("skip_reuse")
            return None
        if pool._affinity_lookup(info):
            self._metric("skip_affinity")
            return None
        # prompt de UMA mensagem repetido: a cadeia nao entra no lookup de afinidade do proxy
        # (range(len-1,0,-1)); confere a cadeia completa direto no mapa
        if info.chain and getattr(pool, "affinity", {}).get(info.chain[-1]):
            self._metric("skip_affinity_full")
            return None
        source = pool.states.get(s.source)
        target = pool.states.get(s.target)
        if source is None or target is None or not source.healthy or not target.healthy:
            self._metric("skip_backend")
            return None
        mgr = target.modes
        if mgr is None or mgr.resolve(info.model) != s.router_model:
            self._metric("skip_model")
            return None
        if source.inflight or source.queued or source.handoff_busy or source.low_inflight:
            self._metric("skip_source_busy")
            return None
        if not mgr.can_acquire_for_handoff(s.router_model):
            self._metric("skip_mode")
            return None
        # "sem cache util no destino": o mapa de afinidade do proxy e a autoridade sobre o dono do
        # prefixo (ja checado acima). O tamanho do KV residente num slot NAO e prova de cache util --
        # um prompt grande de outra sessao no mi50-35b nao ajuda este pedido e bloquearia o handoff
        # para sempre; fica como metrica de contexto (pool_handoff_total{result="cached_target_large"}).
        cached = await self._target_cached(mgr)
        if cached is None:
            self._metric("cache_unknown")
        elif cached >= s.max_cached_tokens:
            self._metric("cached_target_large")
        if self.active:                     # sem await entre checar e reservar
            self._metric("skip_busy")
            return None
        self.active = True
        self._free.clear()
        try:
            return await self._run(request, raw, body, info, source, target, mgr)
        except Exception:
            self._finish()
            raise

    def _finish(self) -> None:
        self.active = False
        self._free.set()

    async def _run(self, request, raw, body, info, source, target, mgr):
        s = self.s
        self.stages = {}
        stages = self.stages
        t0 = time.monotonic()
        filename = "pp35-" + uuid.uuid4().hex + ".bin"
        held, lease, token = [], None, None
        opened = False
        close_task: asyncio.Task | None = None
        self._metric("attempt")
        try:
            source.handoff_busy = True
            for _ in range(source.cfg.np):
                await source.sem.acquire()
                source.inflight += 1
                held.append(source)
            # 1) janela na 3090 (para o 27B, sobe o 35B) em paralelo com a garantia do modo m35
            open_task = asyncio.create_task(self.ctl.call(
                "POST", "/open", {"request_id": filename, "ttl_s": s.open_ttl_s},
                timeout=s.open_timeout_s))
            lease_task = asyncio.create_task(
                mgr.acquire_for_handoff(s.router_model, s.lease_timeout_s))
            opening = asyncio.gather(open_task, lease_task, return_exceptions=True)
            try:
                opened_res, lease = await asyncio.shield(opening)
            except asyncio.CancelledError:
                # a janela pode ter aberto no ctl: espera o resultado (shielded) e fecha no finally
                opened_res, lease = await opening
                if not isinstance(opened_res, BaseException):
                    opened = True
                    token = opened_res.get("token")
                    if lease is not None and not isinstance(lease, BaseException):
                        lease.release()
                self._metric("cancel_open")
                raise
            if isinstance(opened_res, BaseException):
                if lease is not None and not isinstance(lease, BaseException):
                    lease.release()
                    lease = None
                raise Handoff35Error("ctl_open", str(opened_res))
            # a janela JA esta aberta: daqui em diante o 27B so volta pelo finally/_close
            opened = True
            token = opened_res.get("token")
            if isinstance(lease, BaseException):
                raise Handoff35Error("mode_unavailable", str(lease))
            stages["open"] = round(time.monotonic() - t0, 2)
            for k, v in (opened_res.get("steps") or {}).items():
                if isinstance(v, (int, float)):
                    stages["ctl_" + str(k).replace("_ms", "")] = round(v / 1000.0, 2)
            # 2) contrato do destino + tokenizador identico nas duas pontas
            t = time.monotonic()
            await self._target_contract(mgr)
            tokens = await self._tokens(body, source=True)
            stages["tokens"] = round(time.monotonic() - t, 2)
            if len(tokens) < s.min_tokens:
                raise Handoff35Error("below_gate", f"{len(tokens)} < {s.min_tokens}")
            if tokens != await self._tokens(body, source=False):
                raise Handoff35Error("tokenizer_mismatch")
            max_tokens = (body.get("max_tokens") or body.get("max_completion_tokens")
                          or s.default_max_tokens)
            if len(tokens) + int(max_tokens) > s.source_ctx_slot:
                raise Handoff35Error("context", f"{len(tokens)}+{max_tokens} > {s.source_ctx_slot}")
            # 3) prefill + save na 3090 (protocolo de operacao de slot, igual a producao)
            src_slot = await self._source_slot()
            prefix = tokens[:-1]
            budget = s.prefill_budget_s(len(tokens))
            t = time.monotonic()
            await self.ctl.engine("POST", "/completion",
                                  {"prompt": prefix, "n_predict": 0, "cache_prompt": True,
                                   "id_slot": src_slot}, timeout=budget)
            saved = await self._source_operations().execute(
                self._engine_request, "save", src_slot, filename, {})
            stages["prefill_save"] = round(time.monotonic() - t, 2)
            n_saved = saved.get("n_saved", saved.get("n_tokens"))
            if n_saved != len(prefix) or int(saved.get("n_written") or 0) <= 0:
                raise Handoff35Error("save", f"n_saved={n_saved} n_written={saved.get('n_written')}")
            # 4) hardlink para o diretorio do preset do destino (tmpfs compartilhado 220/221/223)
            t = time.monotonic()
            linked = await self.ctl.call("POST", "/link", {"token": token, "filename": filename},
                                        timeout=s.stage_timeout_s)
            if int(linked.get("bytes") or 0) < int(saved.get("n_written") or 0):
                raise Handoff35Error("link", "tamanho do link menor que o salvo")
            stages["transfer"] = round(time.monotonic() - t, 2)
            # 5) restore no mi50-35b do router
            dst_slot = await self._dst_slot(mgr)
            t = time.monotonic()
            restored = await mgr.ctl.slot_restore(s.router_model, dst_slot, filename)
            stages["restore"] = round(time.monotonic() - t, 2)
            n_restored = restored.get("n_restored", restored.get("n_tokens"))
            if n_restored != len(prefix):
                raise Handoff35Error("restore", f"n_restored={n_restored} != {len(prefix)}")
            stages["total"] = round(time.monotonic() - t0, 2)
            self._stages(stages)
            self._metric("ok")
            log.info("handoff35: %s", " ".join(f"{k}={v}" for k, v in stages.items()))
            # 6) a 3090 volta ao 27B enquanto as MI50 decodificam (nao atrasa a resposta)
            close_task = asyncio.ensure_future(self._close(token, "decode", filename, held, source))
            held = []
            payload = json.dumps({**body, "id_slot": dst_slot, "cache_prompt": True,
                                  "model": s.router_model}, ensure_ascii=False).encode()
            from .app import UpstreamDown
            try:
                resp, status, cache_n = await self.app._forward(
                    request, target, "handoff35", payload, handoff_cache=len(prefix))
            except (HandoffError, UpstreamDown) as exc:
                self._metric("decode_failed")
                raise Handoff35Error("decode", type(exc).__name__) from None
            if not str(status).startswith("2"):
                self._metric("decode_status")
            self.app.metrics.decision(target.cfg.name, "handoff35")
            self.app.metrics.request(target.cfg.name, str(status), time.monotonic() - t0)
            if str(status).startswith("2"):
                self.app.pool.record_affinity(info, target.cfg.name)
                self.app.metrics.affinity_result(target.cfg.name, len(prefix), cache_n)
            return resp
        except Handoff35Error as exc:
            self.last_error = str(exc)
            self._metric("fallback")
            self._metric("fallback_" + exc.reason)
            log.info("handoff35: %s -> 35B direto nas MI50", exc)
            return None
        except asyncio.CancelledError:
            self._metric("cancel")
            raise
        except Exception as exc:                       # noqa: BLE001 - qualquer falha cai no direto
            self.last_error = f"{type(exc).__name__}: {exc}"
            self._metric("fallback_error")
            log.warning("handoff35: falha inesperada (%s); 35B direto nas MI50", self.last_error)
            return None
        finally:
            # a janela fecha por qualquer caminho; com a janela aberta o 27B volta SEMPRE
            if close_task is None and opened:
                close_task = asyncio.ensure_future(self._close(token, "abort", filename, held, source))
                held = []
            if held:                       # nada aberto: devolve as permissoes na hora
                for st in held:
                    try:
                        st.inflight -= 1
                        st.sem.release()
                    except (ValueError, AttributeError):
                        pass
                source.handoff_busy = False
            if lease is not None and not isinstance(lease, BaseException):
                lease.release()
                lease = None
            if close_task is None:
                self._finish()

    async def _close(self, token, reason, filename, held, source):
        """Fecha a janela no ctl e so entao devolve as permissoes da 3090 (o 27B esta de pe)."""
        t = time.monotonic()
        res = None
        try:
            res = await self.ctl.call("POST", "/close",
                                      {"token": token, "reason": reason, "filename": filename},
                                      timeout=self.s.close_timeout_s)
            if res.get("ok"):
                # ANTES de liberar a fila curta dos 27B: com o 27B ja servindo, o pedido que
                # esperou vai para a 3090 em vez de forcar m35->m27 no router
                await self._refresh_source_health()
        except CtlError as exc:
            self._metric("close_failed")
            self.last_error = f"close: {exc}"
            log.error("handoff35: falha ao fechar a janela (%s); ctl tem TTL/recover", exc)
        finally:
            for st in held:
                try:
                    st.inflight -= 1
                    st.sem.release()
                except (ValueError, AttributeError):
                    pass
            if source is not None:
                source.handoff_busy = False
            self._finish()
        if res is None:
            return None
        dt = round(time.monotonic() - t, 2)
        self._metric("closed" if res.get("ok") else "close_unrestored")
        steps = {k: v for k, v in (res.get("steps") or {}).items() if isinstance(v, (int, float))}
        self._stages({("close_" + str(k).replace("_ms", "")): (v / 1000.0 if k.endswith("_ms") else v)
                      for k, v in steps.items()})
        self._stages({"close_total": dt})
        if not (res.get("cleanup") or {}).get("absent", False):
            self._metric("cleanup_unproven")
        if not res.get("ok"):
            self.last_error = f"close sem 27B: {res.get('error')}"
            log.error("handoff35: 27B NAO confirmado no close (%s)", res)
        return res

    async def _refresh_source_health(self):
        st = self.app.pool.states.get(self.s.source)
        check = getattr(self.app, "_check", None)
        if st is not None and check is not None:
            try:
                await check(st)
            except Exception:
                log.warning("handoff35: health check da origem falhou apos o close")
