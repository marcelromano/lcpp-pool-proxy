"""Sessoes da faixa low: o KV de cada worker de agente persiste entre turnos no diretorio compartilhado do handoff.

Um worker reenvia o contexto inteiro a cada turno. Com np=1 na 3090 e varios workers se alternando, o slot e
sobrescrito a cada turno; nas MI50 o prefill de dezenas de milhares de tokens custa minutos. Depois de cada turno low
servido por um backend de `low_session_save_from`, o slot e salvo (`/slots/N?action=save`) num arquivo por sessao; no
turno seguinte, se o backend escolhido nao tem o prefixo, o arquivo e restaurado num slot livre antes do pedido (mesmo
contrato de estado do handoff: `handoff_state_format`). Falha em qualquer passo cai no caminho normal (prefill).

Sessao = hash em cadeia ate a 2a mensagem (system + primeira tarefa): o system do worker e igual para todos.
"""
from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import uuid
from collections import OrderedDict
from contextlib import closing
from pathlib import Path
from .handoff import handoff_files_absent
from .slotops import pending_filenames

log = logging.getLogger("pool_proxy.sessions")


class LowSessions:
    def __init__(self, app) -> None:
        self.app = app
        self.s = app.cfg.settings
        self.entries: OrderedDict[str, dict] = OrderedDict()
        self._saving: set[str] = set()
        self._retired_pinned: list[dict] = []

    def _ledger_roots(self) -> set[Path]:
        roots: set[Path] = set()
        if getattr(self.s, "handoff_source_dir", None):
            roots.add(Path(self.s.handoff_source_dir))
        if getattr(self.s, "handoff_target_dir", None):
            roots.add(Path(self.s.handoff_target_dir))
        if getattr(self.s, "slotops_ledger_dir", None):
            roots.add(Path(self.s.slotops_ledger_dir))

        cfg = getattr(self.app, "cfg", None)
        if cfg is not None:
            if getattr(cfg, "slotops_ledger_dir", None):
                roots.add(Path(cfg.slotops_ledger_dir))
            backends = getattr(cfg, "backends", None)
            if backends:
                b_iter = backends.values() if isinstance(backends, dict) else backends
                for b in b_iter:
                    if getattr(b, "slotops_ledger_dir", None):
                        roots.add(Path(b.slotops_ledger_dir))

        pool = getattr(self.app, "pool", None)
        if pool and hasattr(pool, "states"):
            for st in pool.states.values():
                if getattr(st.cfg, "slotops_ledger_dir", None):
                    roots.add(Path(st.cfg.slotops_ledger_dir))

        modes = getattr(self.app, "modes", None)
        if modes and isinstance(modes, dict):
            for m in modes.values():
                if getattr(m, "ctl", None) and getattr(m.ctl, "ledger_dir", None):
                    roots.add(Path(m.ctl.ledger_dir))
                if getattr(m, "ctl", None) and hasattr(m.ctl, "_slot_operations"):
                    for client in m.ctl._slot_operations.values():
                        if hasattr(client, "path"):
                            roots.add(Path(client.path).parent)

        cold_handoff = getattr(self.app, "cold_handoff", None)
        if cold_handoff and hasattr(cold_handoff, "slot_operations"):
            for client in cold_handoff.slot_operations.values():
                if hasattr(client, "path"):
                    roots.add(Path(client.path).parent)

        return roots

    def _pending_filenames(self) -> set[str]:
        return pending_filenames(self._ledger_roots())

    def _purge_orphans(self) -> None:
        """O indice vive em memoria: arquivos sess-* de um processo anterior ficariam orfaos no tmpfs para sempre
        (2026-10-10: 10 arquivos, ~8 GB, levaram /kvslots a 90%). Apaga todos na subida."""
        if not self.enabled():
            return
        try:
            pending = self._pending_filenames()
            if "*" in pending:
                log.warning("sessao: ledger indisponivel; preservando arquivos para reconcile")
                return
            indexed = {e["file"] for e in self.entries.values()} | {e["file"] for e in self._retired_pinned}
            for p in Path(self.s.handoff_source_dir).glob("sess-*"):
                base = p.name if p.name.endswith(".bin") else p.name.split(".bin")[0] + ".bin"
                if base in pending or base in indexed:
                    continue
                p.unlink(missing_ok=True)
        except OSError:
            log.warning("sessao: limpeza de arquivos orfaos falhou")

    def enabled(self) -> bool:
        return bool(self.s.low_session and self.s.handoff_source_dir)

    def _identity(self, st, alias: str) -> dict:
        """Resolve only configured model identities; never use ModeManager's default fallback."""
        cfg = st.cfg
        if getattr(cfg, "router", False):
            manager = getattr(st, "modes", None)
            if manager is None:
                return {"canonical": None, "engine_id": "", "mode_epoch": None, "config_id": None}
            if alias:
                if alias not in manager.known and alias not in manager.aliases:
                    return {"canonical": None, "engine_id": "", "mode_epoch": manager.epoch,
                            "config_id": None}
                canonical = manager.resolve(alias)
            else:
                canonical = self.app.cold_handoff.model_of(st)
            engine_id = cfg.model_engine_ids.get(canonical or "", "")
            mode_epoch = manager.epoch
        else:
            aliases = getattr(cfg, "model_aliases", {})
            if alias in aliases:
                canonical = aliases[alias]
                if not isinstance(canonical, str) or not canonical:
                    return {"canonical": None, "engine_id": "", "mode_epoch": None, "config_id": None}
            else:
                canonical = alias or getattr(cfg, "handoff_model", "") or None
            engine_id = getattr(cfg, "engine_id", "")
            mode_epoch = None
        config_id = None
        path = getattr(cfg, "handoff_gguf_path", "")
        if path:
            try:
                config_id = self.app.cold_handoff.ready_digest(path)
            except OSError:
                # Missing config evidence disables cross-engine restore, but does not
                # prevent a same-engine session from using its exact configured ID.
                config_id = None
        return {"canonical": canonical, "engine_id": engine_id,
                "mode_epoch": mode_epoch, "config_id": config_id}

    @staticmethod
    def sid(info) -> str | None:
        return info.chain[1] if len(info.chain) >= 2 else None

    def lookup(self, info) -> dict | None:
        """Estado salvo cujo conteudo e prefixo desta conversa (mesmas mensagens ate o ponto salvo)."""
        e = self.entries.get(self.sid(info) or "")
        if (e and e.get("model_alias", e.get("model", "")) == getattr(info, "model", "")
                and len(info.chain) > e["nmsgs"] and info.chain[e["nmsgs"] - 1] == e["key"]):
            self.entries.move_to_end(self.sid(info))
            return e
        return None

    def _cross_restore_allowed(self, entry: dict, target, target_identity: dict) -> bool:
        """Prove a configured direct->router compatibility edge without loading either model."""
        states = getattr(getattr(self.app, "pool", None), "states", {})
        origin = entry.get("origin")
        source = states.get(origin)
        if source is None or source is target:
            return False
        source_cfg = source.cfg
        target_cfg = target.cfg
        if (not source_cfg.handoff_state_format
                or source_cfg.handoff_state_format != entry.get("fmt")
                or source_cfg.handoff_state_format != target_cfg.handoff_state_format):
            return False

        alias = entry.get("model_alias", entry.get("model", "")) or ""
        source_identity = self._identity(source, alias)
        if not source_identity.get("engine_id") or entry.get("model_engine_id") != source_identity["engine_id"]:
            return False
        if entry.get("model_canonical") != source_identity.get("canonical"):
            return False
        source_digest = source_identity.get("config_id")
        target_digest = target_identity.get("config_id")
        if not source_digest or source_digest != entry.get("model_config_id") or source_digest != target_digest:
            return False

        source_router = bool(getattr(source_cfg, "router", False))
        target_router = bool(getattr(target_cfg, "router", False))
        if not source_router and not target_router:
            # Two direct engines can share an explicitly configured GGUF and
            # format even when neither exposes a separate canonical name.
            return (source_identity.get("canonical") == entry.get("model_canonical")
                    == target_identity.get("canonical"))

        # The direct source may have been called with an empty model. In that case
        # map the configured public default through router aliases, requiring one
        # unambiguous canonical child. This is metadata/config resolution only;
        # it does not inspect loaded mode state or trigger autoload.
        public_aliases = [alias] if alias else list(getattr(self.app.cfg, "models", ()))
        mapped = set()
        modes = getattr(self.app, "modes", {})
        for public_alias in public_aliases:
            for manager in modes.values():
                try:
                    if public_alias in manager.known or public_alias in manager.aliases:
                        mapped.add(manager.resolve(public_alias))
                except (KeyError, ValueError):
                    continue
        canonical = target_identity.get("canonical")
        if (not canonical or mapped != {canonical}
                or entry.get("model_canonical") not in (None, canonical)):
            return False
        return True

    def pin_lookup(self, info) -> dict | None:
        entry = self.lookup(info)
        if entry is not None:
            entry["pins"] = entry.get("pins", 0) + 1
        return entry

    def unpin(self, entry: dict) -> None:
        entry["pins"] = max(0, entry.get("pins", 0) - 1)
        if entry.get("retired") and not entry["pins"]:
            if entry in self._retired_pinned:
                self._retired_pinned.remove(entry)
            self._unlink(entry)

    def _size(self, filename: str) -> int:
        root = Path(self.s.handoff_source_dir)
        return sum(p.stat().st_size for p in [root / filename, *root.glob(filename + ".*")] if p.exists())

    def _drop(self, e: dict) -> None:
        e["retired"] = True
        if e.get("pins", 0) > 0:
            if e not in self._retired_pinned:
                self._retired_pinned.append(e)
        else:
            self._unlink(e)

    def _clean_persisted_index(self, kind: str, item_key: str, expected_file: str | None = None) -> bool:
        """True only when every persisted ledger was readable and the matching rows are gone (or absent)."""
        pending = self._pending_filenames()
        if "*" in pending:
            return False  # unreadable ledger: delete nothing, removal not proven
        patterns = (
            ".pool-slot-operations-*.sqlite3",
            "pool-slot-operations-*.sqlite3",
            "pool-router-slotops-*.sqlite3",
            ".pool-router-slotops-*.sqlite3",
        )
        seen_dbs = set()
        for root in self._ledger_roots():
            if not root.is_dir():
                continue
            for pat in patterns:
                for db_file in root.glob(pat):
                    try:
                        db_str = str(db_file.resolve())
                    except OSError:
                        db_str = str(db_file)
                    if db_str in seen_dbs:
                        continue
                    seen_dbs.add(db_str)
                    try:
                        with closing(sqlite3.connect(db_file, timeout=5, isolation_level="IMMEDIATE")) as db:
                            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
                            if "snapshot_index" not in tables:
                                continue
                            rows = db.execute(
                                "SELECT owner, metadata FROM snapshot_index WHERE kind=? AND item_key=?",
                                (kind, str(item_key))
                            ).fetchall()
                            for owner, meta_json in rows:
                                should_delete = True
                                if expected_file is not None:
                                    try:
                                        meta = json.loads(meta_json)
                                        if meta.get("file") != expected_file:
                                            should_delete = False
                                    except Exception:
                                        pass
                                if should_delete:
                                    db.execute(
                                        "DELETE FROM snapshot_index WHERE owner=? AND kind=? AND item_key=?",
                                        (owner, kind, str(item_key))
                                    )
                            db.commit()
                    except (sqlite3.Error, OSError):
                        return False
        return True

    def _unlink(self, e: dict) -> bool:
        """Retire a session snapshot: persist the index removal FIRST, then collect the file.
        collected=True only when cleanup returns exactly True AND handoff_files_absent() proves main file + sidecars
        gone. Any other outcome (exception, None, False, True contradicted or unprovable on disk) keeps the entry
        tracked in _retired_pinned (retried by _gc)."""
        collected = False
        try:
            self._remove_indexes(e)
            pending = self._pending_filenames()  # proof: nothing (pin aside) still protects the file
            if "*" not in pending and e["file"] not in pending:
                result = self.app.cold_handoff.cleanup(e["file"])
                if result is True:
                    collected = handoff_files_absent(self.s, e["file"])  # a True claim is still checked against the disk
        except Exception:
            log.warning("sessao: retirada de snapshot nao concluida; mantido rastreado", exc_info=True)
        if not collected and e not in self._retired_pinned:
            self._retired_pinned.append(e)
        return collected

    def _remove_indexes(self, e: dict) -> None:
        if e.get("sid"):
            expected_file = e.get("file")
            cold_ops = getattr(self.app.cold_handoff, "slot_operations", {})
            for client in cold_ops.values():
                client.remove_index("session", e["sid"], expected_file=expected_file)
            modes = getattr(self.app, "modes", {})
            if isinstance(modes, dict):
                for mgr in modes.values():
                    ctl = getattr(mgr, "ctl", None)
                    if ctl and hasattr(ctl, "_slot_operations"):
                        for client in ctl._slot_operations.values():
                            client.remove_index("session", e["sid"], expected_file=expected_file)
            if not self._clean_persisted_index("session", e["sid"], expected_file=expected_file):
                raise RuntimeError("persisted session index removal not proven")

    def _retry_retired(self) -> None:
        for e in [x for x in self._retired_pinned if not x.get("pins")]:
            self._retired_pinned.remove(e)
            self._unlink(e)

    def _free(self) -> int:
        return shutil.disk_usage(self.s.handoff_source_dir).free

    def _gc(self) -> None:
        """Uso REAL: apaga sess-* fora do indice (saves que falharam no meio) e despeja sessoes antigas ate o teto.
        O .ckpt (checkpoints de contexto) pode passar do tamanho do estado; contar so o que o save devolveu estourou o
        tmpfs em 2026-10-10."""
        root = Path(self.s.handoff_source_dir)
        self._retry_retired()
        pending = self._pending_filenames()
        if "*" in pending:
            return
        known = {e["file"] for e in self.entries.values()} | {e["file"] for e in self._retired_pinned} | self._saving | pending
        usage = 0
        for p in root.glob("sess-*"):
            base = p.name if p.name.endswith(".bin") else p.name.split(".bin")[0] + ".bin"
            try:
                if base not in known:
                    p.unlink(missing_ok=True)
                else:
                    usage += p.stat().st_size
            except OSError:
                pass
        while self.entries and usage > self.s.low_session_max_bytes:
            candidate = next(((k, e) for k, e in self.entries.items() if not e.get("pins")), None)
            if candidate is None:
                break
            _k, victim = candidate
            self.entries.pop(_k)
            usage -= self._size(victim["file"])
            self._drop(victim)

    async def save(self, st, info, slot: int) -> dict | None:
        if st.cfg.name not in self.s.low_session_save_from or not st.cfg.handoff_state_format:
            return None
        alias = getattr(info, "model", "") or ""
        identity = self._identity(st, alias)
        if getattr(st.cfg, "router", False):
            model_target = self.app.cold_handoff.model_of(st)
            if not identity["canonical"] or not identity["engine_id"] or identity["canonical"] != model_target:
                log.warning("sessao: router identidade de modelo invalida; fail-closed")
                return None
        if "*" in self._pending_filenames():
            log.warning("sessao: backpressure; ledger ilegivel mantém snapshots protegidos")
            return None
        sid = self.sid(info)
        if sid is None:
            return None
        self._gc()
        while self.entries and self._free() < self.s.low_session_min_free_bytes:
            candidate = next(((k, e) for k, e in self.entries.items() if not e.get("pins")), None)
            if candidate is None:
                break
            _k, victim = candidate
            self.entries.pop(_k)
            self._drop(victim)
        if self._free() < self.s.low_session_min_free_bytes:
            return None            # sem folga no tmpfs: nao arrisca o handoff de producao
        filename = "sess-" + uuid.uuid4().hex + ".bin"
        self._saving.add(filename)
        published = []
        metadata = {"kind": "session", "sid": sid, "key": info.chain[-1], "nmsgs": len(info.chain),
                    "file": filename, "fmt": st.cfg.handoff_state_format, "origin": st.cfg.name,
                    "origin_slot": [st.cfg.name, slot],
                    "model": alias, "model_alias": alias,
                    "model_canonical": identity["canonical"],
                    "model_engine_id": identity["engine_id"],
                    "model_mode_epoch": identity["mode_epoch"],
                    "model_config_id": identity["config_id"]}
        def commit(envelope):
            if envelope.get("status") != "completed":
                self.app.cold_handoff.cleanup(filename)
                return
            published_metadata = {**metadata,
                                  "model_engine_id": envelope.get("engine_id", metadata["model_engine_id"]),
                                  "model_engine_epoch": envelope.get("engine_epoch")}
            entry = self._publish_entry(published_metadata, envelope)
            if entry:
                published.append(entry)
        try:
            await self.app.cold_handoff.req(st, "POST", f"/slots/{slot}?action=save", {"filename": filename},
                                            publish=commit, metadata=metadata)
        finally:
            self._saving.discard(filename)
        if not published:
            self.app.cold_handoff.cleanup(filename)
            return None
        e = published[0]
        self._gc()
        return e if sid in self.entries else None

    def _publish_entry(self, metadata, envelope):
        result = envelope.get("result") or {}
        filename, sid = metadata["file"], metadata["sid"]
        n = int(result.get("n_saved") or 0)
        if n < self.s.low_session_min_tokens or int(result.get("n_written") or 0) <= 0:
            self.app.cold_handoff.cleanup(filename)
            return None
        entry = {"sid": sid, "key": metadata["key"], "nmsgs": int(metadata["nmsgs"]), "file": filename, "tokens": n,
                 "size": self._size(filename), "fmt": metadata["fmt"], "origin": metadata["origin"],
                 "origin_slot": tuple(metadata["origin_slot"]), "pins": 0,
                 "model": metadata.get("model_alias", metadata.get("model", "")),
                 "model_alias": metadata.get("model_alias", metadata.get("model", "")),
                 "model_canonical": metadata.get("model_canonical"),
                 "model_engine_id": metadata.get("model_engine_id"),
                 "model_engine_epoch": metadata.get("model_engine_epoch"),
                 "model_mode_epoch": metadata.get("model_mode_epoch"),
                 "model_config_id": metadata.get("model_config_id")}
        old = self.entries.pop(sid, None)
        if old:
            self._drop(old)
        self.entries[sid] = entry
        return entry

    def recover_operation(self, metadata, envelope):
        """Rebuild the session index from a committed ledger publish before startup ACK."""
        recovered = {**metadata,
                     "model_engine_id": envelope.get("engine_id", metadata.get("model_engine_id")),
                     "model_engine_epoch": envelope.get("engine_epoch", metadata.get("model_engine_epoch"))}
        entry = self._publish_entry(recovered, envelope)
        return entry

    async def restore_into(self, st, e: dict, slot: int) -> bool:
        """Restaura o estado salvo da sessao no slot (reservado pelo chamador)."""
        if st.cfg.handoff_state_format != e["fmt"] or not st.cfg.handoff_state_format:
            return False
        alias = e.get("model_alias", e.get("model", "")) or ""
        identity = self._identity(st, alias)
        if not identity["engine_id"]:
            log.warning("sessao: restore sem engine identity configurada; fail-closed")
            return False
        # Old session rows lack the complete contract. With no explicit migration mapping in
        # this snapshot they remain readable/indexed, but are never sent to an engine.
        if "model_canonical" not in e or "model_engine_id" not in e:
            log.warning("sessao: metadata legado sem allowlist de identidade; fail-closed")
            return False
        if getattr(st.cfg, "router", False):
            model_target = self.app.cold_handoff.model_of(st)
            if not identity["canonical"] or identity["canonical"] != model_target:
                log.warning("sessao: restore router alias/canonical incompatibles; fail-closed")
                return False
        same_engine = e.get("origin") == st.cfg.name
        if same_engine:
            if (e.get("model_canonical") != identity["canonical"]
                    or e.get("model_engine_id") != identity["engine_id"]):
                log.warning("sessao: restore engine/model identity mismatch; fail-closed")
                return False
            expected_engine_epoch = e.get("model_engine_epoch")
            if not expected_engine_epoch:
                log.warning("sessao: restore metadata sem engine epoch; fail-closed")
                return False
            if (identity["mode_epoch"] is not None
                    and e.get("model_mode_epoch") != identity["mode_epoch"]):
                log.warning("sessao: restore epoch mismatch; fail-closed")
                return False
            if getattr(st.cfg, "router", False):
                manager = self.app.modes[st.cfg.name]
                client = manager.ctl.slot_operations_for(identity["canonical"])
                request = manager.ctl.slotop_request(identity["canonical"])
            else:
                client = self.app.cold_handoff.operations_for(st)
                request = lambda method, path, body: self.app.cold_handoff._raw_req(st, method, path, body)
            try:
                capability = await client._get_capability(request)
            except Exception as exc:  # noqa: BLE001 - uncertainty preserves fail-closed behavior
                log.warning("sessao: restore engine epoch indisponivel; fail-closed (%s)", type(exc).__name__)
                return False
            if capability.get("engine_epoch") != expected_engine_epoch:
                log.warning("sessao: restore engine epoch mismatch; fail-closed")
                return False
        else:
            if not self._cross_restore_allowed(e, st, identity):
                log.warning("sessao: restore cross-engine fora da allowlist configurada; fail-closed")
                return False
        r = await self.app.cold_handoff.req(st, "POST", f"/slots/{slot}?action=restore", {"filename": e["file"]})
        return int(r.get("n_restored") or 0) >= self.s.low_session_min_tokens

    def stop(self) -> None:
        for e in list(self.entries.values()):
            self._drop(e)
        self.entries.clear()
        # Pinned retired entries (a live reader holds the bytes) are NOT collected here; they stay tracked.
        # Unpinned ones get one verified attempt; failure leaves them tracked (file kept).
        for e in [x for x in self._retired_pinned if not x.get("pins")]:
            self._retired_pinned.remove(e)
            self._unlink(e)
