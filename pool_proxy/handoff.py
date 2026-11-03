"""Cold text prefill -> slot save -> transport -> restore -> original OAI chat.

The pair is reserved before any await, only while idle. All traffic to the engines
must go through this proxy; /slots cannot provide an atomic external reservation.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import time
import uuid

import aiohttp

from .modes import ModelUnavailable
from .slotops import SlotOperations, SlotOperationError, persisted_operation_identity, validate_terminal_envelope

log = logging.getLogger("pool_proxy.handoff")


def handoff_files_absent(settings, filename: str) -> bool:
    """Proof of absence for a handoff snapshot: True ONLY if every configured root (source/target) enumerates without
    error and neither `filename` nor any `filename.*` sidecar is present. Explicit os.scandir/os.lstat: Path.exists()
    and glob() swallow OSError and would turn an unreadable/missing root into a false "absent". A missing or
    unreadable root, EIO, EACCES, ENOTDIR... => False (not proven). Only FileNotFoundError on the main file lstat,
    under a root that enumerated fine, counts as legitimate absence."""
    if not filename:
        return False
    prefix = filename + "."
    roots = [getattr(settings, "handoff_source_dir", None), getattr(settings, "handoff_target_dir", None)]
    checked = 0
    for directory in set(roots):
        if not isinstance(directory, (str, os.PathLike)) or not os.fspath(directory):
            return False                                 # None/empty/invalid configured root: absence not proven
        checked += 1
        try:
            with os.scandir(directory) as it:           # root missing/unreadable/EIO -> OSError -> not proven
                for entry in it:
                    if entry.name == filename or entry.name.startswith(prefix):
                        return False
        except OSError:
            return False
        try:
            os.lstat(os.path.join(directory, filename))
            return False                                 # the main file exists (appeared after enumeration)
        except FileNotFoundError:
            pass                                         # legitimate absence under a root that enumerated fine
        except OSError:
            return False
    return checked > 0                                   # zero roots verified is not proof of absence


class HandoffError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class ColdHandoff:
    def __init__(self, app):
        self.app = app
        self.s = app.cfg.settings
        self.hashes = {}
        self.hash_tasks = {}
        self.hash_warmer = None
        self.slot_operations = {}
        self._quarantined_holds = {}
        self.quarantine_reconciler = None

    @staticmethod
    def hash_key(path):
        p = Path(path).absolute()  # Preserve /proc/<pid>/root namespace magic links.
        stat = p.stat()
        return (str(p), stat.st_size, stat.st_mtime_ns)

    def ready_digest(self, path):
        """Only inspect metadata/cache here; never await a file read in a request."""
        key = self.hash_key(path)
        for old in list(self.hashes):
            if old[0] == key[0] and old != key:
                del self.hashes[old]
        if key not in self.hashes and key[0] not in self.hash_tasks:
            self.hash_tasks[key[0]] = asyncio.create_task(self._hash_file(key))
        return self.hashes.get(key)

    async def _hash_file(self, key):
        try:
            value = await asyncio.to_thread(self.digest, key[0])
            if self.hash_key(key[0]) == key:
                self.hashes[key] = value
        except (OSError, HandoffError):
            log.warning("handoff GGUF hash unavailable")
        finally:
            self.hash_tasks.pop(key[0], None)

    def client_for_st(self, st):
        if getattr(st.cfg, "router", False):
            manager = self.app.modes.get(st.cfg.name)
            model = self.model_of(st)
            if manager and model:
                return manager.ctl.slot_operations_for(model)
        return self.operations_for(st)

    def start(self):
        if self.s.handoff_enabled:
            self.hash_warmer = asyncio.create_task(self._warm_hashes())
        self.quarantine_reconciler = asyncio.create_task(self._reconcile_quarantined_loop())

    async def _warm_hashes(self):
        while True:
            for name in (self.s.handoff_source, self.s.handoff_target):
                try:
                    self.ready_digest(self.app.pool.states[name].cfg.handoff_gguf_path)
                except OSError:
                    log.warning("handoff GGUF metadata unavailable")
            await asyncio.sleep(1)

    async def _reconcile_quarantined_loop(self):
        while True:
            try:
                for hold_key in list(self._quarantined_holds):
                    await self.resolve_quarantined(hold_key)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning("error in quarantine reconciliation: %s", e)
            await asyncio.sleep(0.5)

    async def stop(self):
        if self.quarantine_reconciler:
            self.quarantine_reconciler.cancel()
            await asyncio.gather(self.quarantine_reconciler, return_exceptions=True)
        if self.hash_warmer:
            self.hash_warmer.cancel()
            await asyncio.gather(self.hash_warmer, return_exceptions=True)
        # to_thread cannot cancel an active read. Drain it before closing the app.
        await asyncio.gather(*list(self.hash_tasks.values()), return_exceptions=True)
        # Stop without verified terminal confirmation or process fence must NEVER clear holds.
        # Only explicitly reconciled operations (or epoch fence) release resources.
        for hold_key, q in list(self._quarantined_holds.items()):
            client = q.get("client")
            op_id = q.get("op_id")
            filename = q.get("filename")
            if client:
                try:
                    with client._db() as db:
                        if op_id:
                            row = db.execute("SELECT state FROM operations WHERE op_id=?", (op_id,)).fetchone()
                        elif filename:
                            row = db.execute("SELECT state FROM operations WHERE filename=?", (filename,)).fetchone()
                        else:
                            row = None
                    if row and row["state"] in ("pending", "running", "ack_pending"):
                        continue
                except Exception:
                    continue

    def operations_for(self, st):
        client = self.slot_operations.get(st.cfg.name)
        if client is None:
            root = Path(st.cfg.slotops_ledger_dir)
            db = root / (".pool-slot-operations-" + hashlib.sha256(st.cfg.name.encode()).hexdigest()[:16] + ".sqlite3")
            client = self.slot_operations[st.cfg.name] = SlotOperations(
                db, "backend:" + st.cfg.name, timeout_s=self.s.handoff_timeout_s,
                engine_id=st.cfg.engine_id, engine_owner=st.cfg.engine_owner,
                endpoint=st.cfg.url, model=st.cfg.handoff_model or "")
        return client

    async def reconcile_startup(self):
        """Run before health/mode workers start; unresolved same-epoch rows keep the app unready."""
        for st in self.app.pool.states.values():
            if st.cfg.router:
                manager = self.app.modes.get(st.cfg.name)
                if manager is None:
                    continue
                models = sorted({value[0] for value in st.cfg.modes.values()}
                                | manager.ctl.pending_operation_models())
                for model in models:
                    request = manager.ctl.slotop_request(model)
                    client = manager.ctl.slot_operations_for(model)
                    await client.reconcile_startup(
                        request, self.cleanup,
                        publish=lambda metadata, result, backend=st: self.recover_published_operation(
                            backend, metadata, result))
                continue
            else:
                request = lambda method, path, body: self._raw_req(st, method, path, body)
                client = self.operations_for(st)
            await client.reconcile_startup(
                request, self.cleanup,
                publish=lambda metadata, result: self.recover_published_operation(st, metadata, result))

    @staticmethod
    def model_of(st):
        """Modelo canonico do router que recebe o handoff, ou None em engine direta."""
        return st.cfg.handoff_model if st.cfg.router else None

    def recover_published_operation(self, st, metadata, result):
        if metadata.get("kind") == "session":
            self.app.sessions.recover_operation(metadata, result)
        elif metadata.get("kind") == "mode":
            manager = self.app.modes.get(st.cfg.name)
            if manager:
                manager.recover_operation(metadata, result)

    async def req(self, st, method, path, body=None, routed=True, *, publish=None, metadata=None, timeout=None):
        """Em backend router o modelo vai no query (GET) ou no corpo (POST) e autoload=false: nunca
        carregar por acidente. `routed=False` e para a gerencia do proprio router (GET /models).
        `timeout` sobrepoe handoff_timeout_s por chamada (o prefill da origem usa o orcamento de
        Settings.handoff_prefill_budget_s; a operacao de slot no router mantem o timeout do ledger)."""
        from .app import _auth
        op = re.fullmatch(r"/slots/(\d+)\?action=(save|restore|erase)", path)
        if method == "POST" and op:
            filename = (body or {}).get("filename", "")
            if op.group(2) != "erase" and (not isinstance(filename, str) or not filename):
                raise HandoffError("snapshot filename missing")
            model = self.model_of(st)
            if model:
                manager = self.app.modes[st.cfg.name]
                client = manager.ctl.slot_operations_for(model)
                operation_request = manager.ctl.slotop_request(model)
            else:
                client = self.operations_for(st)

                async def operation_request(op_method, op_path, op_body):
                    return await self._raw_req(st, op_method, op_path, op_body, routed=routed)
            try:
                return await client.execute(operation_request, op.group(2), int(op.group(1)), filename,
                                            {k: v for k, v in (body or {}).items() if k != "filename"},
                                            publish=publish, metadata=metadata)
            except Exception as exc:
                raise HandoffError(f"terminal slot operation failed closed: {exc}",
                                   getattr(exc, "status", None)) from exc
        return await self._raw_req(st, method, path, body, routed=routed, timeout=timeout)

    async def _raw_req(self, st, method, path, body=None, routed=True, timeout=None):
        from .app import _auth
        model, params = self.model_of(st), None
        if model and routed:
            params = {"autoload": "false"}
            if method == "GET":
                params["model"] = model
            else:
                body = {**(body or {}), "model": model}
        async with self.app.session.request(
                method, st.cfg.url.rstrip("/") + path, json=body, params=params,
                headers=_auth(st), timeout=aiohttp.ClientTimeout(total=timeout or self.s.handoff_timeout_s)) as r:
            if r.status < 200 or r.status >= 300:
                # Never include upstream bodies, paths, prompts or credentials in logs.
                raise HandoffError(f"HTTP {r.status}", status=r.status)
            if r.status == 204:
                return {}
            return await r.json()

    def digest(self, path):
        key = self.hash_key(path)
        h = hashlib.sha256()
        with Path(key[0]).open("rb") as f:
            before = os.fstat(f.fileno())
            for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
                h.update(chunk)
            after = os.fstat(f.fileno())
        if (before.st_size, before.st_mtime_ns) != key[1:] or (
                after.st_size, after.st_mtime_ns) != key[1:] or self.hash_key(path) != key:
            raise HandoffError("GGUF changed while hashing")
        return h.hexdigest()

    async def tokens(self, st, body):
        body = {k: v for k, v in body.items() if k != "model"} if self.model_of(st) else body
        templated = await self.req(st, "POST", "/apply-template", body)
        prompt = templated.get("prompt")
        if not isinstance(prompt, str):
            raise HandoffError("non-text template")
        result = await self.req(st, "POST", "/tokenize", {
            "content": prompt, "add_special": True, "parse_special": True})
        tokens = result.get("tokens")
        if not isinstance(tokens, list) or not tokens or any(type(t) is not int for t in tokens):
            raise HandoffError("invalid tokens")
        return tokens

    async def contract(self, st):
        """Destino router: o preset do modelo carregado tem de declarar cada opcao de `handoff_preset`
        (K/V, kv-unified: o arquivo de slot carrega n_stream e os tipos de cache; divergir falha no restore)."""
        want = st.cfg.handoff_preset
        if not want:
            return
        data = (await self.req(st, "GET", "/models", routed=False)).get("data") or []
        entry = next((m for m in data if m.get("id") == self.model_of(st)), None)
        status = (entry or {}).get("status") or {}
        if status.get("value") not in ("loaded", "sleeping"):
            raise HandoffError("router model not loaded")
        preset = {}
        for line in str(status.get("preset", "")).splitlines():
            k, sep, v = line.partition("=")
            if sep:
                preset[k.strip()] = v.strip().lower()
        for k, v in want.items():
            if preset.get(k) != str(v).lower():
                raise HandoffError(f"router preset contract mismatch ({k})")

    async def slot(self, st):
        slots = await self.req(st, "GET", "/slots")
        if not isinstance(slots, list) or len(slots) != st.cfg.np:
            raise HandoffError("slot capacity mismatch")
        idle = [s for s in slots if s.get("is_processing") is False and not s.get("has_media", False)
                and type(s.get("id")) is int and 0 <= s["id"] < st.cfg.np]
        if not idle:
            raise HandoffError("no idle text slot")
        return min(idle, key=lambda s: s.get("n_cached_tokens", s.get("n_prompt_tokens", 0)))["id"]

    async def source_slot(self, st, requested=None):
        # Candidate the caller must own every source permit throughout
        # selection, prefill and save. handoff_busy excludes proxy admissions.
        if not st.handoff_busy or st.inflight != st.cfg.np or st.queued:
            raise HandoffError("source is not exclusively reserved")
        slots = await self.req(st, "GET", "/slots")
        if (not isinstance(slots, list) or len(slots) != st.cfg.np
                or {s.get("id") for s in slots} != set(range(st.cfg.np))
                or any(s.get("is_processing") is not False or s.get("has_media", False)
                       for s in slots)):
            raise HandoffError("source slots not idle text slots")
        if requested is not None:
            if type(requested) is not int or requested not in range(st.cfg.np):
                raise HandoffError("invalid source slot")
            return requested
        # Reuse the slot that owns the resident KV instead of selecting the
        # empty slot and provoking an attention allocation failure + purge.
        def resident(s):
            kv = s.get("kv_capacity", {})
            n = kv.get("seq_cells") if kv.get("known") else None
            if type(n) is not int or n < 0:
                n = s.get("n_prompt_tokens", s.get("n_cached_tokens", 0))
            return n if type(n) is int and n >= 0 else 0
        return max(slots, key=lambda s: (resident(s), -s["id"]))["id"]

    async def _source_leg(self, source, filename, prefix, src_slot, stages, timeout=None):
        """Prefill do prefixo (sem o ultimo token) na origem e save do slot; devolve o tamanho gravado.
        `timeout` e o orcamento do prefill (Settings.handoff_prefill_budget_s), nao handoff_timeout_s:
        a 130k tokens o prefill sozinho passa de handoff_timeout_s."""
        t = time.monotonic()
        try:
            return await self._source_leg_run(source, filename, prefix, src_slot, timeout)
        finally:
            stages["prefill_save"] = time.monotonic() - t

    async def _source_leg_run(self, source, filename, prefix, src_slot, timeout=None):
        await self.req(source, "POST", "/completion", {
            "prompt": prefix, "n_predict": 0, "cache_prompt": True, "id_slot": src_slot}, timeout=timeout)
        saved = await self.req(source, "POST", f"/slots/{src_slot}?action=save", {"filename": filename},
                               timeout=timeout)
        if saved.get("n_saved") != len(prefix) or saved.get("n_written", 0) <= 0:
            raise HandoffError("incomplete save")
        return saved["n_written"]

    async def quiesce(self, st, slot=None, timeout=None):
        """Depois de um preparo cancelado o /completion da origem pode continuar ocupando o slot por instantes.
        O caminho normal escolhe slot por LCP similarity: se o slot ainda estiver `is_processing`, o engine escolhe
        OUTRO slot e apaga o prefixo parcial que ja estava quente (foi o que aconteceu em uma rodada
        normal refez 130k tokens frios). Espera, limitado, o slot sair de processing. True = livre. Nunca levanta."""
        budget = self.s.handoff_quiesce_s if timeout is None else timeout
        if budget <= 0:
            return False
        end = time.monotonic() + budget
        while True:
            try:
                slots = await self.req(st, "GET", "/slots")
            except Exception:                          # noqa: BLE001 - melhor cair no caminho normal
                return False
            if isinstance(slots, list):
                busy = [s for s in slots if isinstance(s, dict) and s.get("is_processing")
                        and (slot is None or s.get("id") == slot)]
                if not busy:
                    return True
            if time.monotonic() >= end:
                return False
            await asyncio.sleep(0.2)

    async def _lease(self, acquire, box, stages, t0):
        box.append(await acquire())
        stages["lease"] = time.monotonic() - t0

    async def prepare(self, source, target, body, filename, acquire=None, box=None, stages=None, dst_slot=None,
                      src_slot=None, progress=None):
        """`acquire` (destino router) devolve a lease do modelo do handoff e ela vai para `box`.
        Com `handoff_switch_overlap` a lease (e portanto a carga do modelo) corre DURANTE o prefill da origem:
        so as verificacoes que precisam do destino (props, contrato, tokenizador, slot, restore) esperam por ela.
        Sem sobreposicao a lease vem antes de qualquer chamada a origem."""
        stages = {} if stages is None else stages
        clock = time.monotonic
        overlap = acquire is not None and self.s.handoff_switch_overlap
        lease_task = None
        t = clock()
        if acquire is not None and not overlap:
            await self._lease(acquire, box, stages, t)
        elif overlap and not target.modes.can_acquire_for_handoff(self.model_of(target)):
            # politica nega: falhar antes de gastar o prefill da origem
            raise ModelUnavailable(f"{target.cfg.name}: politica nao permite trocar para {self.model_of(target)}")
        checked = (source,) if overlap else (source, target)
        for st in checked:
            props = await self.req(st, "GET", "/props")
            if props.get("model_path") != st.cfg.handoff_model_path:
                raise HandoffError("loaded GGUF mismatch")
            await self.contract(st)
        # These read-only mounts must be the files loaded at the /props paths.
        hashes = [self.ready_digest(st.cfg.handoff_gguf_path) for st in (source, target)]
        if any(h is None for h in hashes):
            raise HandoffError("GGUF hash not ready")
        if hashes[0] != hashes[1]:
            raise HandoffError("GGUF SHA256 mismatch")
        t1 = clock()
        tokens = await self.tokens(source, body)
        stages["tokens"] = clock() - t1
        if len(tokens) < max(2, self.s.handoff_min_tokens):
            raise HandoffError("below exact threshold")
        for st in (source, target):
            max_tokens = body.get("max_tokens") or body.get("max_completion_tokens") or self.s.default_max_tokens
            if st.cfg.ctx_max and len(tokens) + max_tokens > st.cfg.ctx_max:
                raise HandoffError("context capacity")
        if not overlap:
            if tokens != await self.tokens(target, body):
                raise HandoffError("template/tokenizer mismatch")
        fixed = dst_slot
        src_slot = await self.source_slot(source, src_slot)
        dst_slot = fixed if fixed is not None or overlap else await self.slot(target)
        # Orcamento do prefill: derivado do numero EXATO de tokens e da taxa medida do backend de origem.
        # handoff_timeout_s sozinho nao cobre 130k tokens (207 s a 629 t/s e 305 s a 423 t/s) — .
        prefill_budget = self.s.handoff_prefill_budget_s(len(tokens), source.cfg.prefill_tps)
        if progress is not None:                 # diagnostico fora de `stages` (stages vira metrica por etapa)
            progress["src_slot"] = src_slot
            progress["prefill_budget"] = round(prefill_budget, 2)
        # Keep the final prompt token unevaluated. A restored recurrent state
        # cannot rewind one token just to recover logits for an identical prompt;
        # the target must append that token instead (TAG_PROMPT_LOGITS upstream).
        prefix = tokens[:-1]
        t2 = clock()
        leg = asyncio.ensure_future(self._source_leg(source, filename, prefix, src_slot, stages, prefill_budget))
        if overlap:
            lease_task = asyncio.ensure_future(self._lease(acquire, box, stages, t2))
        # Sem return_exceptions a primeira falha liberaria o chamador com a outra perna ainda rodando.
        # Se a lease falha, o prefill termina mesmo assim: a origem fica com o prefixo quente para o fallback.
        async with asyncio.timeout(prefill_budget):
            results = await asyncio.gather(leg, *([lease_task] if lease_task else []), return_exceptions=True)
        for r in results:
            if isinstance(r, BaseException):
                raise r
        written = results[0]
        if overlap:
            # tempo que o prefill NAO escondeu da troca (0 = troca totalmente sobreposta)
            # At centisecond resolution, sub-10ms event-loop scheduling noise reports as complete overlap.
            stages["lease_wait"] = round(max(0.0, stages["lease"] - stages["prefill_save"]), 2)
            props = await self.req(target, "GET", "/props")
            if props.get("model_path") != target.cfg.handoff_model_path:
                raise HandoffError("loaded GGUF mismatch")
            await self.contract(target)
            if tokens != await self.tokens(target, body):
                raise HandoffError("template/tokenizer mismatch")
            dst_slot = fixed if fixed is not None else await self.slot(target)
        t3 = clock()
        await self.transfer(filename, written)
        stages["transfer"] = clock() - t3
        t4 = clock()
        restored = await self.req(target, "POST", f"/slots/{dst_slot}?action=restore", {"filename": filename})
        stages["restore"] = clock() - t4
        if restored.get("n_restored") != len(prefix):
            raise HandoffError("incomplete restore")
        return dst_slot, len(tokens)

    async def transfer(self, filename, size):
        src, dst = Path(self.s.handoff_source_dir), Path(self.s.handoff_target_dir)
        main = src / filename
        if main.stat().st_size != size:
            raise HandoffError("save file size mismatch")
        files = [main, *src.glob(filename + ".*")]
        # Transfer all siblings: .dft, .ckpt, future metadata.
        if self.s.handoff_transport == "copy":
            for p in files:
                out = dst / p.name
                if p.resolve() != out.resolve():
                    with p.open("rb") as source, out.open("xb") as target:
                        while chunk := source.read(1024 * 1024):
                            target.write(chunk)
                            await asyncio.sleep(0)
        elif not (dst / filename).exists() or not main.samefile(dst / filename):
            raise HandoffError("shared directories do not refer to the same file")

    def is_quarantined_file(self, filename: str) -> bool:
        if not filename:
            return False
        sessions = getattr(self.app, "sessions", None)
        if sessions and hasattr(sessions, "_pending_filenames"):
            pending = sessions._pending_filenames()
            if "*" in pending or filename in pending:
                return True
        if any(h.get("filename") == filename for h in self._quarantined_holds.values()):
            return True
        for client in self.slot_operations.values():
            if client.is_quarantined_file(filename):
                return True
        modes = getattr(self.app, "modes", {})
        if isinstance(modes, dict):
            for mgr in modes.values():
                ctl = getattr(mgr, "ctl", None)
                if ctl and hasattr(ctl, "_slot_operations"):
                    for client in ctl._slot_operations.values():
                        if client.is_quarantined_file(filename):
                            return True
        return False

    def retain_quarantine(self, st, filename: str, held: list, box: list, client, op_id: str | None = None):
        permits = [s for s in held if s == st]
        for s in permits:
            held.remove(s)
        leases = list(box)
        box.clear()
        # A filename is a resource protected by one or more operations, not an
        # operation identity. Keep concurrent/retried operations independently
        # addressable even when they reference the same file.
        hold_key = f"{st.cfg.name}:{op_id or uuid.uuid4().hex}"
        row = None
        if client:
            try:
                with client._db() as db:
                    if op_id:
                        row = db.execute(
                            "SELECT * FROM operations WHERE op_id=?", (op_id,)
                        ).fetchone()
                    else:
                        rows = db.execute(
                            "SELECT * FROM operations WHERE filename=? AND state IN "
                            "('pending','completed','failed','cancelled_before_start','published','ack_pending')",
                            (filename,)
                        ).fetchall()
                        if len(rows) == 1:
                            row = rows[0]
            except Exception:
                row = None
        op_tuple = (persisted_operation_identity(
            row, owner=client.owner, engine_id=client.engine_id,
            engine_owner=client.engine_owner, endpoint=client.endpoint, model=client.model)
                    if row and client else None)
        hold_token = None
        if op_tuple and client:
            try:
                hold_token = client.register_caller_hold(op_tuple["op_id"], op_tuple)
            except Exception as exc:
                # Preserve resources in memory; an unregistered hold cannot be retired
                # from row absence or an ACK proof it does not own.
                log.warning("caller hold registration failed; resources remain quarantined: %s", exc)
        self._quarantined_holds[hold_key] = {
            "st": st,
            "permits": permits,
            "leases": leases,
            "client": client,
            "filename": filename,
            "op_id": op_tuple["op_id"] if op_tuple else op_id,
            "op_tuple": op_tuple,
            "hold_token": hold_token,
        }
        self._refresh_backend_busy(st)

    def quarantined_operation_id(self, client, filename: str) -> str | None:
        """Resolve a unique filename to its complete persisted operation identity."""
        if client is None or not filename:
            return None
        try:
            with client._db() as db:
                rows = db.execute(
                    "SELECT * FROM operations WHERE owner=? AND filename=? AND state IN "
                    "('pending','completed','failed','cancelled_before_start','published','ack_pending')",
                    (client.owner, filename),
                ).fetchall()
        except Exception:
            return None
        if len(rows) != 1:
            return None
        identity = persisted_operation_identity(
            rows[0], owner=client.owner, engine_id=client.engine_id,
            engine_owner=client.engine_owner, endpoint=client.endpoint, model=client.model)
        if identity is None or identity.get("filename") != filename:
            return None
        return identity["op_id"]

    def _refresh_backend_busy(self, st):
        """Keep the shared backend admission guard set while any operation owns a hold."""
        st.handoff_busy = any(q.get("st") is st for q in self._quarantined_holds.values())

    async def resolve_quarantined(self, hold_key: str, request_fn=None):
        q = self._quarantined_holds.get(hold_key)
        if not q:
            return
        st = q["st"]
        client = q["client"]
        filename = q["filename"]
        target_op_id = q.get("op_id")
        expected = q.get("op_tuple")
        # Older or corrupt rows do not contain enough evidence to release a
        # resource. Missing identity is UNKNOWN, including during shutdown.
        required = ("op_id", "owner", "engine_id", "engine_epoch", "owner_seq", "action",
                    "filename", "endpoint", "model", "slot")
        if (not target_op_id or not isinstance(expected, dict)
                or any(k not in expected or expected[k] is None for k in required)
                or any(not isinstance(expected[k], str) or not expected[k] for k in
                       ("op_id", "owner", "engine_id", "engine_epoch", "action", "endpoint"))
                or not isinstance(expected["model"], str)
                or type(expected["owner_seq"]) is not int or expected["owner_seq"] < 1
                or type(expected["slot"]) is not int or expected["slot"] < 0
                or not isinstance(expected["filename"], str)):
            return
        if (expected["op_id"] != target_op_id or expected["engine_id"] != client.engine_id
                or expected["endpoint"] != client.endpoint or expected["model"] != client.model):
            return
        request = request_fn
        if request is None:
            if getattr(st.cfg, "router", False):
                manager = self.app.modes.get(st.cfg.name)
                model = self.model_of(st)
                if manager and model:
                    request = manager.ctl.slotop_request(model)
            if request is None:
                request = lambda method, path, body: self._raw_req(st, method, path, body)
        try:
            cap = await client._get_capability(request)
        except Exception:
            return

        try:
            with client._db() as db:
                rows = db.execute("SELECT * FROM operations WHERE op_id=?", (target_op_id,)).fetchall()
        except Exception:
            # Ledger corrupt or unreadable: hold must remain active (fail-closed)
            return

        # A successful independent execute may ACK and remove its operation row
        # before this caller hold is reconciled. Only its persisted exact proof can
        # substitute for that row; absence alone remains UNKNOWN.
        if len(rows) == 0:
            proof = client.caller_ack_proof(target_op_id, expected, q.get("hold_token"))
            if (proof and proof.get("identity") == expected
                    and isinstance(proof.get("acked_at"), (int, float))
                    and cap.get("engine_id") == expected["engine_id"]
                    and cap.get("owner") == expected["owner"]
                    and validate_terminal_envelope(
                        proof.get("terminal"), op_id=expected["op_id"],
                        engine_id=expected["engine_id"], engine_epoch=expected["engine_epoch"],
                        owner=expected["owner"], owner_seq=expected["owner_seq"],
                        action=expected["action"], slot=expected["slot"],
                        filename=expected["filename"])):
                status = proof["terminal"]["status"]
                self._release_quarantined(hold_key, require_proof=True)
                if status != "completed":
                    self.cleanup(filename)
            return
        if len(rows) != 1:
            return
        row = rows[0]

        actual = persisted_operation_identity(
            row, owner=client.owner, engine_id=client.engine_id,
            engine_owner=client.engine_owner, endpoint=client.endpoint, model=client.model)
        if actual != expected or cap.get("engine_id") != expected["engine_id"] or cap.get("owner") != expected["owner"]:
            return

        # Verifiable process fence stale cleanup: if engine epoch bumped, stale process died
        if row["epoch"] != cap["engine_epoch"]:
            with client._db() as db:
                db.execute("UPDATE operations SET state='stale_epoch' WHERE op_id=?", (row["op_id"],))
            self._release_quarantined(hold_key)
            self.cleanup(filename)
            return

        try:
            resp = await request("GET", f"/slot-operations/{row['op_id']}", None)
        except Exception:
            return

        slot_val = expected["slot"]
        if validate_terminal_envelope(
            resp,
            op_id=row["op_id"],
            engine_id=client.engine_id,
            engine_epoch=row["epoch"],
            owner=expected["owner"],
            owner_seq=row["seq"],
            action=row["action"],
            slot=slot_val,
            filename=filename,
        ):
            with client._db() as db:
                db.execute("UPDATE operations SET state=?, result=? WHERE op_id=?",
                           (resp["status"], json.dumps(resp), row["op_id"]))
            try:
                await client._ack(request, row["op_id"], row["seq"], row["epoch"], expected["owner"])
            except Exception:
                # A terminal read alone does not retire caller-owned resources.
                # Keep the row, writer quarantine, permits, and leases for retry.
                return
            if resp["status"] != "completed":
                self._release_quarantined(hold_key, require_proof=True)
            else:
                self._release_quarantined(hold_key, require_proof=True)

    def _release_quarantined(self, hold_key: str, *, require_proof: bool = False):
        q = self._quarantined_holds.get(hold_key)
        if not q:
            return
        client = q.get("client")
        last_reference = False
        if client and q.get("op_tuple"):
            token = q.get("hold_token")
            if not token:
                raise SlotOperationError("caller hold has no registered resource reference")
            # This transaction must succeed before caller resources are released.
            # Errors propagate so reconciliation retries with the hold still intact.
            last_reference = client.retire_caller_hold(
                q["op_tuple"]["op_id"], q["op_tuple"], token,
                require_proof=require_proof)
        q = self._quarantined_holds.pop(hold_key, None)
        if not q:
            return
        st = q["st"]
        for lease in q.get("leases", []):
            try:
                lease.release()
            except Exception:
                pass
        for permit in q.get("permits", []):
            try:
                st.inflight -= 1
                st.sem.release()
            except Exception:
                pass
        # A backend may have several independently identified holds. Releasing
        # one operation must not open admission while another still owns permits.
        self._refresh_backend_busy(st)
        if last_reference:
            self.cleanup_retired_handoff(q.get("filename", ""))

    def cleanup_retired_handoff(self, filename: str) -> bool:
        """Delete only an abandoned transient handoff after durable last-ref retirement.
        Returns the verified result of cleanup(); False for every no-op/skip (callers historically ignore it)."""
        if not re.fullmatch(r"pp-handoff-[0-9a-f]{32}\.bin", filename or ""):
            return False
        sessions = getattr(self.app, "sessions", None)
        if not sessions or not hasattr(sessions, "_pending_filenames"):
            return False
        try:
            pending = sessions._pending_filenames()
        except Exception:
            return False
        if "*" in pending or filename in pending:
            return False
        if self.is_quarantined_file(filename):
            return False
        return self.cleanup(filename)

    def cleanup(self, filename: str) -> bool:
        """Remove the snapshot and its sidecars. Returns True ONLY when, after the attempt, the main file and every
        sidecar are verifiably absent from the source and target dirs. False = nothing proven: empty name, ledger
        unknown ('*'), quarantine/hold/pending protection, unlink OSError, or a file still present."""
        if not filename:
            return False
        sessions = getattr(self.app, "sessions", None)
        if sessions and hasattr(sessions, "_pending_filenames"):
            if "*" in sessions._pending_filenames():
                return False
        if self.is_quarantined_file(filename):
            return False
        for directory in {self.s.handoff_source_dir, self.s.handoff_target_dir}:
            root = Path(directory)
            try:
                files = [root / filename, *root.glob(filename + ".*")]
            except OSError:
                log.warning("handoff temporary directory cleanup failed")
                continue
            for p in files:
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    log.warning("handoff temporary file cleanup failed")
        return handoff_files_absent(self.s, filename)

    async def attempt(self, request, raw, body, info):
        s, pool = self.s, self.app.pool
        if (not s.handoff_enabled or info.model not in s.handoff_models or info.has_image
                or "id_slot" in body or info.est_prompt_tokens < s.handoff_min_tokens
                or max(info.reuse.values(), default=0) > 0 or info.hint
                or pool._affinity_lookup(info)):
            return None
        source, target = pool.states[s.handoff_source], pool.states[s.handoff_target]
        eligible = pool._eligible(info, set())
        model = self.model_of(target)
        if model and target.modes.resolve(info.model) != model:
            self.app.metrics.handoffs["wrong_model"] += 1   # pedido de outro modelo do router
            return None
        if any(st not in eligible or st.inflight or st.queued or st.handoff_busy for st in (source, target)):
            return None
        try:
            hashes = [self.ready_digest(st.cfg.handoff_gguf_path) for st in (source, target)]
        except OSError:
            hashes = [None]
        if any(h is None for h in hashes):
            self.app.metrics.handoffs["hash_not_ready"] += 1
            return None
        # Atomic in the event loop: no await between checking and reserving the pair.
        for st in (source, target):
            st.handoff_busy = True
        held, box, stages = [], [], {}
        progress: dict = {}
        filename = "pp-handoff-" + uuid.uuid4().hex + ".bin"
        t0 = time.monotonic()
        source_quarantined = False
        target_quarantined = False
        try:
            for st in (source, target):
                for _ in range(st.cfg.np):
                    # With inflight=queued=0 all permits are free. Semaphore.acquire
                    # completes without yielding, unlike Pool.acquire's wait_for wrapper.
                    await st.sem.acquire()
                    st.inflight += 1
                    held.append(st)
            try:
                acquire = None
                if model:
                    # modelo carregado antes de qualquer restore; a lease impede a troca de modo
                    # (drain) enquanto o decode roda
                    def acquire():
                        return target.modes.acquire_for_handoff(model, s.handoff_timeout_s)
                # Sem timeout externo unico: cada await de prepare() tem limite proprio (chamada HTTP =
                # handoff_timeout_s; prefill+save = Settings.handoff_prefill_budget_s derivado dos tokens
                # EXATOS; lease = acquire_for_handoff). O teto unico de handoff_timeout_s garantia a falha
                # de todo handoff acima de ~115k tokens na origem.
                slot, count = await self.prepare(source, target, body, filename, acquire, box, stages,
                                                 progress=progress)
            except ModelUnavailable as e:
                log.info("handoff: destino sem o modelo (%s); roteamento normal", e)
                self.app.metrics.handoffs["model_unavailable"] += 1
                return None
            except BaseException as e:
                if not isinstance(e, (asyncio.CancelledError, KeyboardInterrupt)):
                    log.info("handoff: preparo falhou (%s: %s); roteamento normal", type(e).__name__, e)
                self.app.metrics.handoffs["fallback"] += 1
                source_client = self.client_for_st(source)
                target_client = self.client_for_st(target)
                if source_client and source_client.is_quarantined_file(filename):
                    source_quarantined = True
                    source_op_id = self.quarantined_operation_id(source_client, filename)
                    # The destination mode lease was acquired alongside the source
                    # save in overlap mode. Transfer it with the exact source op hold;
                    # finally must not release it while that writer is unresolved.
                    self.retain_quarantine(source, filename, held, box, source_client, source_op_id)
                if target_client and target_client.is_quarantined_file(filename):
                    target_quarantined = True
                    self.retain_quarantine(target, filename, held, box, target_client)
                if not source_quarantined and progress.get("src_slot") is not None and "prefill_save" in stages:
                    # O /completion cancelado pode ainda ocupar o slot da origem; o caminho normal escolhe
                    # slot por LCP similarity e, com o slot ocupado, apaga o outro prefixo quente (rodada 3
                    # de  refez 130k tokens frios). Espera limitada; nunca falha o pedido.
                    if await self.quiesce(source, progress.get("src_slot")):
                        log.info("handoff: slot %s da origem livre para o caminho normal", progress.get("src_slot"))
                if isinstance(e, (asyncio.CancelledError, KeyboardInterrupt)):
                    raise
                return None
            self.app.metrics.handoff_stages(stages)
            log.info("handoff: etapas %s", " ".join(f"{k}={v:.2f}s" for k, v in stages.items()))
            # The 3090 becomes available while the MI50 decodes.
            while source in held:
                held.remove(source)
                pool.release(source, None)
            self._refresh_backend_busy(source)
            payload = {**body, "id_slot": slot, "cache_prompt": True}
            if model:
                payload["model"] = model
            payload = json.dumps(payload, ensure_ascii=False).encode()
            self.app.metrics.decision(target.cfg.name, "handoff")
            from .app import UpstreamDown
            try:
                resp, status, cache_n = await self.app._forward(
                    request, target, "handoff", payload, handoff_cache=count - 1)
            except HandoffError:
                self.app.metrics.handoffs["fallback"] += 1
                return None
            except UpstreamDown:
                pool.mark_failure(target, hard=True)
                self.app.metrics.handoffs["fallback"] += 1
                return None
            self.app.metrics.handoffs["ok" if status.startswith("2") else "upstream_error"] += 1
            self.app.metrics.handoffs["cache_unknown" if cache_n is None else
                                     "cache_hit" if cache_n >= count - 1 else "cache_miss"] += 1
            if status.startswith("2"):
                pool.record_affinity(info, target.cfg.name)
                self.app.metrics.affinity_result(target.cfg.name, count - 1, cache_n)
            self.app.metrics.request(target.cfg.name, status, time.monotonic() - t0)
            return resp
        finally:
            if not target_quarantined:
                for extra in list(box):
                    extra.release()
            for st in list(held):
                pool.release(st, None)
            self._refresh_backend_busy(source)
            self._refresh_backend_busy(target)
            if not source_quarantined and not target_quarantined:
                self.cleanup(filename)
