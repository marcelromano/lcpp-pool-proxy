"""Fail-closed client for the P0 engine slot-operation protocol.

The local ledger is process-restart state, not a power-loss durability promise.
Engine operations remain authoritative and are queried until terminal/ACK.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import time
import uuid
import uuid
import glob
import inspect
from contextlib import contextmanager
from pathlib import Path


class SlotOperationError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, outcome: str = "unknown"):
        super().__init__(message)
        self.status = status
        # Only these explicit outcomes prove that continuing cannot race an
        # accepted writer. Missing/malformed identity and same-epoch 404 stay unknown.
        self.outcome = outcome


TERMINAL = {"completed", "failed", "cancelled_before_start"}
CAPABILITY = "op_id-terminal-v1"


def validate_terminal_envelope(
    resp: dict,
    *,
    op_id: str,
    engine_id: str,
    engine_epoch: str,
    owner: str,
    owner_seq: int,
    action: str,
    slot: int,
    filename: str,
) -> bool:
    """Strict common envelope validator before terminal / ACK / release.
    Every required envelope field must be present and exactly match.
    No mandatory field is optional.
    """
    if not isinstance(resp, dict):
        return False
    required_keys = ("op_id", "engine_id", "engine_epoch", "owner", "owner_seq", "action", "slot", "status")
    for k in required_keys:
        if k not in resp or resp[k] is None:
            return False
    if (any(not isinstance(resp[k], str) or not resp[k] for k in
            ("op_id", "engine_id", "engine_epoch", "owner", "action", "status"))
            or type(resp["owner_seq"]) is not int or resp["owner_seq"] < 1
            or type(resp["slot"]) is not int or resp["slot"] < 0):
        return False
    if resp.get("status") not in TERMINAL:
        return False
    if resp["op_id"] != op_id:
        return False
    if resp["engine_id"] != engine_id:
        return False
    if resp["engine_epoch"] != engine_epoch:
        return False
    if resp["owner"] != owner:
        return False
    if resp["owner_seq"] != owner_seq:
        return False
    if resp["action"] != action:
        return False
    if resp["slot"] != int(slot):
        return False
    # Filename is part of the identity for every action (erase uses "").
    if "filename" not in resp or not isinstance(resp["filename"], str) or resp["filename"] != filename:
        return False
    return True


def persisted_operation_identity(row, *, owner: str, engine_id: str, engine_owner: str,
                                 endpoint: str, model: str) -> dict | None:
    """Return a complete, self-consistent persisted identity or None (fail closed)."""
    try:
        payload = json.loads(row["payload"])
        identity = {
            "op_id": row["op_id"], "owner": row["engine_owner"],
            "engine_id": row["engine_id"], "engine_epoch": row["epoch"],
            "owner_seq": row["seq"], "action": row["action"],
            "slot": payload["slot"], "filename": row["filename"] or "",
            "endpoint": row["endpoint"], "model": row["model"],
        }
        if (not all(isinstance(identity[k], str) and identity[k] for k in
                    ("op_id", "owner", "engine_id", "engine_epoch", "action", "endpoint"))
                or not isinstance(identity["model"], str)
                or type(identity["owner_seq"]) is not int or identity["owner_seq"] < 1
                or type(identity["slot"]) is not int or identity["slot"] < 0
                or not isinstance(identity["filename"], str)
                or identity["action"] not in ("save", "restore", "erase")
                or payload.get("action") != identity["action"]
                or payload.get("filename", "") != identity["filename"]
                or row["owner"] != owner or identity["engine_id"] != engine_id
                or identity["owner"] != engine_owner or identity["endpoint"] != endpoint
                or identity["model"] != model):
            return None
        return identity
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None

# The engine has one strict sequence per authenticated (engine_id, owner), even
# when local callers use separate ledgers (direct/low/router). Serialize those
# callers in this proxy process; the durable per-caller ledgers are reconciled
# before readiness after restart.
_identity_locks: dict[tuple[str, str], asyncio.Lock] = {}


from typing import Iterable


def pending_filenames(directory: str | Path | Iterable[str | Path] = ()) -> set[str]:
    """Files owned by an unacknowledged operation or persisted snapshot index must survive startup and GC."""
    protected: set[str] = set()
    roots: list[Path] = []
    if isinstance(directory, (str, Path)):
        if str(directory):
            roots.append(Path(directory))
    elif hasattr(directory, "__iter__"):
        for d in directory:
            if isinstance(d, (str, Path)) and str(d):
                roots.append(Path(d))

    patterns = (
        ".pool-slot-operations-*.sqlite3",
        "pool-slot-operations-*.sqlite3",
        "pool-router-slotops-*.sqlite3",
        ".pool-router-slotops-*.sqlite3",
    )
    seen_dbs: set[str] = set()
    for root in roots:
        try:
            if not root.is_dir():
                continue
        except OSError:
            return {"*"}
        for pat in patterns:
            try:
                matches = list(root.glob(pat))
            except OSError:
                return {"*"}
            for db_file in matches:
                try:
                    db_str = str(db_file.resolve())
                except OSError:
                    db_str = str(db_file)
                if db_str in seen_dbs:
                    continue
                seen_dbs.add(db_str)
                try:
                    db = sqlite3.connect(db_file)
                    try:
                        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
                        if "operations" in tables:
                            rows = db.execute(
                                "SELECT filename FROM operations WHERE state IN "
                                "('pending','completed','failed','cancelled_before_start','published','ack_pending')"
                            ).fetchall()
                            protected.update(row[0] for row in rows if row[0])
                        if "snapshot_index" in tables:
                            rows = db.execute("SELECT metadata FROM snapshot_index").fetchall()
                            for row in rows:
                                try:
                                    metadata = json.loads(row[0])
                                    if metadata.get("file"):
                                        protected.add(metadata["file"])
                                except Exception:
                                    pass
                        if "caller_ack_hold_refs" in tables:
                            rows = db.execute(
                                "SELECT identity FROM caller_ack_hold_refs").fetchall()
                            for row in rows:
                                try:
                                    identity = json.loads(row[0])
                                    filename = identity["filename"]
                                    if not isinstance(filename, str):
                                        return {"*"}
                                    if filename:
                                        protected.add(filename)
                                except Exception:
                                    # A ref with unknown identity may own any file.
                                    return {"*"}
                    finally:
                        db.close()
                except (sqlite3.Error, OSError):
                    # An unreadable ledger is not evidence that its writer is gone.
                    return {"*"}
    return protected


class SlotOperations:
    def __init__(self, path: str | Path, owner: str, *, poll_s: float = .1,
                 timeout_s: float = 60.0, engine_id: str = "", engine_owner: str = "",
                 endpoint: str = "", model: str = "") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.owner = owner
        # `owner` is a local ledger namespace. `engine_id` is the separately configured
        # identity asserted by the engine endpoint and must never be inferred from it.
        self.engine_id = engine_id
        self.engine_owner = engine_owner
        self.endpoint = endpoint
        self.model = model
        self.poll_s, self.timeout_s = poll_s, timeout_s
        self._lock = asyncio.Lock()
        with self._db() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS meta (
                owner TEXT PRIMARY KEY, epoch TEXT, seq INTEGER NOT NULL DEFAULT 0)""")
            db.execute("""CREATE TABLE IF NOT EXISTS operations (
                op_id TEXT PRIMARY KEY, owner TEXT NOT NULL, seq INTEGER NOT NULL,
                epoch TEXT NOT NULL, action TEXT NOT NULL, payload TEXT NOT NULL,
                payload_hash TEXT NOT NULL, filename TEXT, state TEXT NOT NULL,
                result TEXT, metadata TEXT, updated REAL NOT NULL)""")
            columns = {r["name"] for r in db.execute("PRAGMA table_info(operations)")}
            if "metadata" not in columns:
                db.execute("ALTER TABLE operations ADD COLUMN metadata TEXT")
            if "engine_id" not in columns:
                db.execute("ALTER TABLE operations ADD COLUMN engine_id TEXT")
            if "engine_owner" not in columns:
                db.execute("ALTER TABLE operations ADD COLUMN engine_owner TEXT")
            if "endpoint" not in columns:
                db.execute("ALTER TABLE operations ADD COLUMN endpoint TEXT")
            if "model" not in columns:
                db.execute("ALTER TABLE operations ADD COLUMN model TEXT")
            db.execute("INSERT OR IGNORE INTO meta(owner, seq) VALUES (?, 0)", (owner,))
            db.execute("""CREATE TABLE IF NOT EXISTS snapshot_index (
                owner TEXT NOT NULL, kind TEXT NOT NULL, item_key TEXT NOT NULL,
                metadata TEXT NOT NULL, result TEXT NOT NULL,
                PRIMARY KEY(owner,kind,item_key))""")
            db.execute("""CREATE TABLE IF NOT EXISTS identity_history (
                op_id TEXT PRIMARY KEY, endpoint TEXT NOT NULL, model TEXT NOT NULL,
                engine_id TEXT NOT NULL, engine_owner TEXT NOT NULL, seq INTEGER NOT NULL)""")
            # 's unnamespaced table is retained as unknown legacy evidence. New
            # code never treats it as releasable proof or deletes it by inference.
            db.execute("""CREATE TABLE IF NOT EXISTS caller_ack_proofs_ns (
                namespace_key TEXT NOT NULL, op_id TEXT NOT NULL,
                identity TEXT NOT NULL, terminal TEXT NOT NULL, acked_at REAL NOT NULL,
                PRIMARY KEY(namespace_key,op_id))""")
            db.execute("""CREATE TABLE IF NOT EXISTS caller_ack_hold_refs (
                namespace_key TEXT NOT NULL, op_id TEXT NOT NULL, hold_token TEXT NOT NULL,
                identity TEXT NOT NULL, created_at REAL NOT NULL,
                PRIMARY KEY(namespace_key,op_id,hold_token))""")

    def _db(self):
        @contextmanager
        def connection():
            db = sqlite3.connect(self.path, timeout=5, isolation_level="IMMEDIATE")
            db.row_factory = sqlite3.Row
            try:
                yield db
                db.commit()
            except BaseException:
                db.rollback()
                raise
            finally:
                db.close()
        return connection()

    def _caller_namespace_key(self, expected: dict) -> str | None:
        required = ("op_id", "owner", "engine_id", "engine_epoch", "owner_seq",
                    "action", "slot", "filename", "endpoint", "model")
        if (not isinstance(expected, dict)
                or any(key not in expected or expected[key] is None for key in required)
                or any(not isinstance(expected[key], str) or not expected[key]
                       for key in ("op_id", "owner", "engine_id", "engine_epoch",
                                   "action", "endpoint"))
                or not isinstance(expected["model"], str)
                or type(expected["owner_seq"]) is not int or expected["owner_seq"] < 1
                or type(expected["slot"]) is not int or expected["slot"] < 0
                or not isinstance(expected["filename"], str)
                or expected["owner"] != self.engine_owner
                or expected["engine_id"] != self.engine_id
                or expected["endpoint"] != self.endpoint
                or expected["model"] != self.model):
            return None
        namespace = {"namespace": self.owner, "engine_owner": self.engine_owner,
                     "engine_id": self.engine_id, "endpoint": self.endpoint,
                     "model": self.model}
        return json.dumps(namespace, sort_keys=True, separators=(",", ":"))

    def register_caller_hold(self, op_id: str, expected: dict, hold_token: str | None = None) -> str:
        """Persist one resource reference under its complete caller/engine namespace."""
        namespace_key = self._caller_namespace_key(expected)
        if not namespace_key or expected.get("op_id") != op_id:
            raise SlotOperationError("caller hold namespace or operation identity is incomplete")
        token = hold_token or str(uuid.uuid4())
        with self._db() as db:
            row = db.execute("SELECT * FROM operations WHERE op_id=?", (op_id,)).fetchone()
            if row:
                actual = persisted_operation_identity(
                    row, owner=self.owner, engine_id=self.engine_id,
                    engine_owner=self.engine_owner, endpoint=self.endpoint, model=self.model)
            else:
                proof = db.execute(
                    "SELECT identity FROM caller_ack_proofs_ns WHERE namespace_key=? AND op_id=?",
                    (namespace_key, op_id)).fetchone()
                try:
                    actual = json.loads(proof["identity"]) if proof else None
                except (TypeError, ValueError, json.JSONDecodeError):
                    actual = None
            if actual != expected:
                raise SlotOperationError("caller hold operation identity mismatch")
            db.execute("INSERT OR IGNORE INTO caller_ack_hold_refs VALUES(?,?,?,?,?)",
                       (namespace_key, op_id, token, json.dumps(expected, sort_keys=True), time.time()))
        return token

    def caller_ack_proof(self, op_id: str, expected: dict, hold_token: str) -> dict | None:
        namespace_key = self._caller_namespace_key(expected)
        if not namespace_key or expected.get("op_id") != op_id or not hold_token:
            return None
        try:
            with self._db() as db:
                ref = db.execute(
                    "SELECT identity FROM caller_ack_hold_refs WHERE namespace_key=? AND op_id=? AND hold_token=?",
                    (namespace_key, op_id, hold_token)).fetchone()
                row = db.execute(
                    "SELECT identity,terminal,acked_at FROM caller_ack_proofs_ns "
                    "WHERE namespace_key=? AND op_id=?", (namespace_key, op_id)).fetchone()
            if not ref or not row:
                return None
            identity = json.loads(row["identity"])
            ref_identity = json.loads(ref["identity"])
            terminal = json.loads(row["terminal"])
            if (identity != expected or ref_identity != expected
                    or type(row["acked_at"]) not in (int, float)
                    or not validate_terminal_envelope(
                        terminal, op_id=expected["op_id"], engine_id=expected["engine_id"],
                        engine_epoch=expected["engine_epoch"], owner=expected["owner"],
                        owner_seq=expected["owner_seq"], action=expected["action"],
                        slot=expected["slot"], filename=expected["filename"])):
                return None
            return {"identity": identity, "terminal": terminal, "acked_at": row["acked_at"]}
        except (sqlite3.Error, OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def retire_caller_hold(self, op_id: str, expected: dict, hold_token: str,
                           *, require_proof: bool = False) -> bool:
        """Retire one known reference; return true only when the last proof ref is removed."""
        namespace_key = self._caller_namespace_key(expected)
        if not namespace_key or expected.get("op_id") != op_id or not hold_token:
            raise SlotOperationError("caller hold retirement lacks complete identity")
        encoded = json.dumps(expected, sort_keys=True)
        with self._db() as db:
            ref = db.execute(
                "SELECT identity FROM caller_ack_hold_refs WHERE namespace_key=? AND op_id=? AND hold_token=?",
                (namespace_key, op_id, hold_token)).fetchone()
            if not ref or ref["identity"] != encoded:
                raise SlotOperationError("caller hold reference missing or identity mismatch")
            proof = db.execute(
                "SELECT identity FROM caller_ack_proofs_ns WHERE namespace_key=? AND op_id=?",
                (namespace_key, op_id)).fetchone()
            if proof and proof["identity"] != encoded:
                raise SlotOperationError("caller ACK proof identity mismatch during retirement")
            if require_proof and not proof:
                raise SlotOperationError("caller ACK proof missing during resource retirement")
            db.execute("DELETE FROM caller_ack_hold_refs WHERE namespace_key=? AND op_id=? AND hold_token=?",
                       (namespace_key, op_id, hold_token))
            refs_left = db.execute(
                "SELECT COUNT(*) FROM caller_ack_hold_refs WHERE namespace_key=? AND op_id=?",
                (namespace_key, op_id)).fetchone()[0]
            if refs_left == 0 and proof:
                db.execute("DELETE FROM caller_ack_proofs_ns WHERE namespace_key=? AND op_id=? AND identity=?",
                           (namespace_key, op_id, encoded))
                return True
        return False

    async def _get_capability(self, request):
        try:
            response = await request("GET", "/slot-operations", None)
        except SlotOperationError:
            raise
        except Exception as exc:
            status = getattr(exc, "status", None)
            raise SlotOperationError("slot operation capability unavailable", status) from exc
        if (not isinstance(response, dict) or response.get("capability") != CAPABILITY
                or not isinstance(response.get("engine_epoch"), str) or not response["engine_epoch"]
                or type(response.get("next_seq")) is not int or response["next_seq"] < 1
                or not isinstance(response.get("owner"), str) or not response["owner"]):
            raise SlotOperationError("slot operation capability malformed or unsupported")
        if not self.engine_id or response.get("engine_id") != self.engine_id:
            raise SlotOperationError(f"slot operation engine_id mismatch (configured={self.engine_id!r}, received={response.get('engine_id')!r})")
        if not self.engine_owner or response.get("owner") != self.engine_owner:
            raise SlotOperationError(f"slot operation owner mismatch (configured={self.engine_owner!r}, received={response.get('owner')!r})")
        return response

    async def _refresh_same_epoch(self, request, epoch):
        deadline = time.monotonic() + min(1.0, self.timeout_s)
        while time.monotonic() < deadline:
            try:
                cap = await asyncio.wait_for(self._get_capability(request),
                                             max(0.001, deadline - time.monotonic()))
            except SlotOperationError:
                await asyncio.sleep(min(self.poll_s, max(0.0, deadline - time.monotonic())))
                continue
            except asyncio.TimeoutError:
                return None
            return cap["engine_epoch"] == epoch
        return None

    async def _prove_preaccept_rejection(self, request, op_id, initial_cap) -> bool:
        """Prove no acceptance via exact authenticated epoch/owner/sequence plus absent op id."""
        try:
            cap = await asyncio.wait_for(self._get_capability(request), min(1.0, self.timeout_s))
        except (SlotOperationError, asyncio.TimeoutError):
            return False
        if (cap["engine_epoch"] != initial_cap["engine_epoch"]
                or cap["owner"] != initial_cap["owner"]
                or cap["next_seq"] != initial_cap["next_seq"]):
            return False
        try:
            await asyncio.wait_for(request("GET", f"/slot-operations/{op_id}", None),
                                   min(1.0, self.timeout_s))
        except Exception as exc:
            return getattr(exc, "status", None) == 404
        return False

    def _record(self, epoch, next_seq, action, slot, filename, body, metadata=None, engine_owner=None):
        canonical = {"action": action, "slot": int(slot), "filename": filename, "payload": body}
        payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(payload.encode()).hexdigest()
        with self._db() as db:
            pending = db.execute("SELECT COUNT(*) FROM operations WHERE owner=? AND state IN "
                                 "('pending','completed','failed','cancelled_before_start','published','ack_pending')",
                                 (self.owner,)).fetchone()[0]
            if pending:
                raise SlotOperationError("owner already has an unacknowledged slot operation")
            meta = db.execute("SELECT epoch,seq FROM meta WHERE owner=?", (self.owner,)).fetchone()
            # Engine epoch is a process fence. An epoch change invalidates old pending records;
            # they are quarantined by the ledger and are never published as successful.
            if meta["epoch"] and meta["epoch"] != epoch:
                db.execute("UPDATE operations SET state='stale_epoch', updated=? "
                           "WHERE owner=? AND epoch!=? AND state!='acked'",
                           (time.time(), self.owner, epoch))
            seq = max((meta["seq"] + 1) if meta["epoch"] == epoch else 1, next_seq)
            op_id = str(uuid.uuid4())
            db.execute("UPDATE meta SET epoch=?,seq=? WHERE owner=?", (epoch, seq, self.owner))
            db.execute("INSERT INTO operations (op_id,owner,seq,epoch,action,payload,payload_hash,filename,state,result,metadata,updated,engine_id,engine_owner,endpoint,model) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (op_id, self.owner, seq, epoch, action, payload, digest, filename, "pending", None,
                        json.dumps(metadata, sort_keys=True) if metadata else None, time.time(),
                        self.engine_id, engine_owner, self.endpoint, self.model))
            db.execute("INSERT INTO identity_history VALUES(?,?,?,?,?,?)",
                       (op_id, self.endpoint, self.model, self.engine_id, engine_owner or "", seq))
        return op_id, seq, digest, payload

    def _pending(self, epoch, action, slot, filename, body):
        with self._db() as db:
            rows = db.execute("SELECT * FROM operations WHERE owner=? AND state IN "
                              "('pending','completed','failed','cancelled_before_start','published','ack_pending')",
                              (self.owner,)).fetchall()
        same_epoch = [r for r in rows if r["epoch"] == epoch]
        if not same_epoch:
            return None
        if len(same_epoch) != 1:
            raise SlotOperationError("multiple pending operations require manual reconciliation")
        row = same_epoch[0]
        canonical = json.dumps({"action": action, "slot": int(slot), "filename": filename,
                                "payload": body}, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        if row["action"] != action or row["filename"] != filename or row["payload_hash"] != digest:
            raise SlotOperationError("pending operation must reconcile before another snapshot")
        if row["engine_id"] != self.engine_id:
            raise SlotOperationError("pending operation belongs to another engine; quarantined")
        if persisted_operation_identity(
                row, owner=self.owner, engine_id=self.engine_id, engine_owner=self.engine_owner,
                endpoint=self.endpoint, model=self.model) is None:
            raise SlotOperationError("pending operation identity mismatch; quarantined")
        return row["op_id"], row["seq"], digest, canonical, row["state"], row["result"], row["engine_owner"]

    async def execute(self, request, action: str, slot: int, filename: str,
                      body: dict | None = None, *, publish=None, metadata=None) -> dict:
        """Submit, poll, validate and ACK. Shielding leaves the operation running if caller cancels."""
        task = asyncio.create_task(self._execute(request, action, slot, filename, body or {}, publish, metadata))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # A second disconnect/shutdown cancellation must not unwind the caller's finally block
            # while the engine callback still owns its slot. Drain the finish task exactly once.
            current = asyncio.current_task()
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    if current is not None:
                        current.uncancel()
            try:
                task.result()
            except BaseException:
                pass
            raise asyncio.CancelledError

    async def _execute(self, request, action, slot, filename, body, publish, metadata):
        identity = (self.engine_id, self.engine_owner)
        shared_lock = _identity_locks.setdefault(identity, asyncio.Lock())
        async with shared_lock, self._lock:
            cap = await self._get_capability(request)
            recovered = self._pending(cap["engine_epoch"], action, slot, filename, body)
            if recovered:
                op_id, seq, digest, canonical, local_state, saved_result, row_owner = recovered
                if row_owner != cap["owner"]:
                    raise SlotOperationError("authenticated engine owner changed; operation remains quarantined")
                if local_state == "ack_pending":
                    result = json.loads(saved_result) if saved_result else None
                    canonical_identity = json.loads(canonical)
                    if not validate_terminal_envelope(
                            result, op_id=op_id, engine_id=self.engine_id,
                            engine_epoch=cap["engine_epoch"], owner=cap["owner"],
                            owner_seq=seq, action=action, slot=int(canonical_identity["slot"]),
                            filename=canonical_identity["filename"]):
                        raise SlotOperationError("pending ACK terminal envelope mismatch")
                    if result and result.get("status") == "completed" and publish:
                        value = publish(result)
                        if inspect.isawaitable(value):
                            await value
                    await self._ack(request, op_id, seq, cap["engine_epoch"], cap["owner"])
                    if not result or result.get("status") != "completed":
                        raise SlotOperationError("reconciled operation was terminal failure")
                    return result["result"]
            else:
                op_id, seq, digest, canonical = self._record(
                    cap["engine_epoch"], cap["next_seq"], action, slot, filename, body, metadata, cap["owner"])
            # Match server_routes::handle_slots_{save,restore,erase}: action is the
            # query parameter, slot is the URL, and only save/restore carry filename.
            submitted = {"op_id": op_id, "engine_epoch": cap["engine_epoch"],
                         "owner_seq": seq, "owner": cap["owner"],
                         "engine_id": self.engine_id}
            if action in ("save", "restore"):
                submitted["filename"] = filename
            # The engine binds and hashes the canonical action/slot/file/payload. Retries reuse
            # this identical identity; an ambiguous submission is queried, never replaced.
            endpoint = f"/slots/{int(slot)}?action={action}"
            ambiguous_submit = False
            try:
                result = await request("POST", endpoint, submitted)
            except Exception as exc:
                status = getattr(exc, "status", None)
                if isinstance(status, int) and (400 <= status < 500 or status == 501):
                    if status == 501 and await self._prove_preaccept_rejection(request, op_id, cap):
                        self._reject(op_id, seq, cap["next_seq"])
                        raise SlotOperationError("slot operation unsupported before acceptance", status,
                                                 "preaccept_rejected") from exc
                    if status != 501 and await self._prove_preaccept_rejection(request, op_id, cap):
                        self._reject(op_id, seq, cap["next_seq"])
                        raise SlotOperationError(f"slot operation rejected before acceptance ({status})", status,
                                                 "preaccept_rejected") from exc
                    self._state(op_id, "pending", None)
                    raise SlotOperationError(f"slot operation submission outcome unknown ({status})", status) from exc
                fenced = await self._refresh_same_epoch(request, cap["engine_epoch"])
                if fenced is False:
                    self._state(op_id, "stale_epoch", None)
                    raise SlotOperationError("authenticated same-engine owner epoch fence; cold prefill required",
                                             outcome="fenced") from exc
                if fenced is None:
                    self._state(op_id, "pending", None)
                    raise SlotOperationError("submission outcome unknown; capability could not be verified") from exc
                ambiguous_submit = True
                result = None
            while not isinstance(result, dict) or result.get("status") not in TERMINAL:
                querying = result is not None
                try:
                    if querying:
                        result = await request("GET", f"/slot-operations/{op_id}", None)
                    else:
                        result = await request("POST", endpoint, submitted)
                except Exception as exc:
                    status = getattr(exc, "status", None)
                    if isinstance(status, int) and (400 <= status < 500 or status == 501):
                        if not querying and not ambiguous_submit:
                            if await self._prove_preaccept_rejection(request, op_id, cap):
                                self._reject(op_id, seq, cap["next_seq"])
                                raise SlotOperationError(f"slot operation rejected before acceptance ({status})",
                                                         status, "preaccept_rejected") from exc
                            self._state(op_id, "pending", None)
                            raise SlotOperationError(f"slot operation retry outcome unknown ({status})", status) from exc
                        self._state(op_id, "pending", None)
                        raise SlotOperationError(f"same-epoch operation query rejected ({status})", status) from exc
                    # A transport/5xx outage is ambiguous. Refresh epoch before retrying the exact
                    # tuple; never submit an old-epoch operation to a restarted engine.
                    fenced = await self._refresh_same_epoch(request, cap["engine_epoch"])
                    if fenced is False:
                        self._state(op_id, "stale_epoch", None)
                        raise SlotOperationError("authenticated same-engine owner epoch fence; cold prefill required",
                                                 outcome="fenced") from exc
                    if fenced is None:
                        self._state(op_id, "pending", None)
                        raise SlotOperationError("operation outcome unknown; capability could not be verified") from exc
                    if not querying:
                        ambiguous_submit = True
                    result = None
                if isinstance(result, dict) and result.get("status") in TERMINAL:
                    break
                self._state(op_id, "pending", result)
                await asyncio.sleep(self.poll_s)
            if not validate_terminal_envelope(
                result,
                op_id=op_id,
                engine_id=self.engine_id,
                engine_epoch=cap["engine_epoch"],
                owner=cap["owner"],
                owner_seq=seq,
                action=action,
                slot=int(slot),
                filename=filename,
            ):
                self._state(op_id, "pending", result)
                raise SlotOperationError("slot operation terminal envelope mismatch")
            if result["status"] != "completed":
                self._state(op_id, result["status"], result)
                if publish:
                    value = publish(result)
                    if inspect.isawaitable(value):
                        await value
                self._state(op_id, "ack_pending", result)
                await self._ack(request, op_id, seq, cap["engine_epoch"], cap["owner"])
                raise SlotOperationError(f"slot operation {result['status']}", outcome="terminal_failed")
            self._state(op_id, "completed", result)
            expected = result.get("result")
            if not isinstance(expected, dict):
                raise SlotOperationError("completed operation has no result")
            if action == "save" and (type(expected.get("n_written")) is not int or expected["n_written"] <= 0):
                raise SlotOperationError("completed save lacks verified byte count")
            if action == "restore" and type(expected.get("n_restored")) is not int:
                raise SlotOperationError("completed restore lacks token count")
            if action == "erase" and type(expected.get("n_erased")) is not int:
                raise SlotOperationError("completed erase lacks terminal result")
            # Persist terminal first, then commit the proxy index/publish point before ACK.
            if publish:
                value = publish(result)
                if inspect.isawaitable(value):
                    await value
            self._state(op_id, "published", result)
            self._state(op_id, "ack_pending", result)
            await self._ack(request, op_id, seq, cap["engine_epoch"], cap["owner"])
            return expected

    async def _ack(self, request, op_id, seq, epoch, owner, *, deadline=None):
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                raise SlotOperationError("bounded reconcile deadline: exact terminal ACK remains pending")
            try:
                ack = request("POST", f"/slot-operations/{op_id}/ack",
                              {"engine_epoch": epoch, "owner_seq": seq,
                               "owner": owner, "engine_id": self.engine_id})
                if deadline is None:
                    await ack
                else:
                    await asyncio.wait_for(ack, max(0.001, deadline - time.monotonic()))
                self._state(op_id, "acked", None)
                return
            except Exception as exc:
                status = getattr(exc, "status", None)
                if isinstance(status, int) and 400 <= status < 500:
                    try:
                        cap = await self._get_capability(request)
                    except SlotOperationError:
                        await asyncio.sleep(self.poll_s)
                        continue
                    if cap["engine_epoch"] != epoch:
                        self._state(op_id, "stale_epoch", None)
                        return
                if deadline is not None and time.monotonic() >= deadline:
                    raise SlotOperationError("bounded reconcile deadline: exact terminal ACK remains pending") from exc
            if deadline is not None and time.monotonic() >= deadline:
                raise SlotOperationError("bounded reconcile deadline: exact terminal ACK remains pending")
            sleep_for = self.poll_s
            if deadline is not None:
                sleep_for = min(sleep_for, max(0.0, deadline - time.monotonic()))
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)

    def _reject(self, op_id, seq, engine_next_seq):
        """A pre-accept 4xx must not create a local sequence gap."""
        with self._db() as db:
            db.execute("DELETE FROM operations WHERE op_id=?", (op_id,))
            db.execute("UPDATE meta SET seq=? WHERE owner=?", (max(0, engine_next_seq - 1), self.owner))

    async def reconcile_startup(self, request, cleanup=None, publish=None, timeout_s=None):
        """Reconcile durable rows. Optional deadline leaves unresolved rows intact."""
        deadline = time.monotonic() + timeout_s if timeout_s is not None else None
        async def bounded_request(method, path, body):
            if deadline is None:
                return await request(method, path, body)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SlotOperationError("bounded reconcile deadline expired before transport request")
            try:
                return await asyncio.wait_for(request(method, path, body), remaining)
            except asyncio.TimeoutError as exc:
                raise SlotOperationError("bounded reconcile transport request exceeded deadline") from exc
        with self._db() as db:
            # Unknown caller ownership survives restart; absence from an in-memory
            # map is not authority to retire an ACK proof or resource reference.
            rows = db.execute("SELECT * FROM operations WHERE owner=? AND state IN "
                              "('pending','completed','failed','cancelled_before_start','published','ack_pending') "
                              "ORDER BY updated", (self.owner,)).fetchall()
        if not rows:
            if publish:
                for metadata, result in self.published_entries():
                    value = publish(metadata, result)
                    if inspect.isawaitable(value):
                        await value
            return
        cap = await self._get_capability(bounded_request)
        for row in rows:
            identity = persisted_operation_identity(
                row, owner=self.owner, engine_id=self.engine_id, engine_owner=cap["owner"],
                endpoint=self.endpoint, model=self.model)
            if identity is None:
                raise SlotOperationError("startup persisted operation identity mismatch; backend remains unready")
            if identity["engine_epoch"] != cap["engine_epoch"]:
                self._state(row["op_id"], "stale_epoch", None)
                if cleanup and row["filename"]:
                    cleanup(row["filename"])
                continue
            result = json.loads(row["result"]) if row["result"] else None
            if row["state"] in ("pending", "completed", "failed", "cancelled_before_start"):
                while not isinstance(result, dict) or result.get("status") not in TERMINAL:
                    if deadline is not None and time.monotonic() >= deadline:
                        raise SlotOperationError("bounded reconcile still has an accepted nonterminal operation")
                    try:
                        result = await bounded_request("GET", f"/slot-operations/{row['op_id']}", None)
                    except Exception as exc:
                        if getattr(exc, "status", None) == 404:
                            raise SlotOperationError("same-epoch operation is unknown; backend remains unready", 404) from exc
                        await asyncio.sleep(min(self.poll_s, max(0.0, deadline - time.monotonic()))
                                            if deadline is not None else self.poll_s)
                        continue
                    if not isinstance(result, dict) or result.get("status") not in TERMINAL:
                        if deadline is not None and time.monotonic() >= deadline:
                            raise SlotOperationError("bounded reconcile still has an accepted nonterminal operation")
                        await asyncio.sleep(min(self.poll_s, max(0.0, deadline - time.monotonic()))
                                            if deadline is not None else self.poll_s)
                canonical = json.loads(row["payload"])
                if not validate_terminal_envelope(
                    result,
                    op_id=row["op_id"],
                    engine_id=self.engine_id,
                    engine_epoch=row["epoch"],
                    owner=cap["owner"],
                    owner_seq=row["seq"],
                    action=row["action"],
                    slot=int(canonical.get("slot", 0)),
                    filename=canonical.get("filename", "") or (row["filename"] or ""),
                ):
                    raise SlotOperationError("startup reconcile envelope mismatch")
                if result["status"] == "completed":
                    expected = result.get("result")
                    if not isinstance(expected, dict):
                        raise SlotOperationError("startup completed operation has no result")
                    if row["action"] == "save" and (type(expected.get("n_written")) is not int or expected["n_written"] <= 0):
                        raise SlotOperationError("startup completed save lacks verified byte count")
                    if row["action"] == "restore" and type(expected.get("n_restored")) is not int:
                        raise SlotOperationError("startup completed restore lacks token count")
                    if row["action"] == "erase" and type(expected.get("n_erased")) is not int:
                        raise SlotOperationError("startup completed erase lacks terminal result")
                metadata = json.loads(row["metadata"]) if row["metadata"] else None
                if result["status"] == "completed" and metadata:
                    self._commit_index(metadata, result)
                elif result["status"] == "completed" and row["action"] == "save" and row["filename"] and cleanup:
                    # No durable publish metadata: this staging snapshot was never visible.
                    cleanup(row["filename"])
                elif result["status"] != "completed" and row["filename"] and cleanup:
                    cleanup(row["filename"])
                self._state(row["op_id"], "ack_pending", result)
            elif row["state"] in ("published", "ack_pending") and result and result.get("status") == "completed":
                if not validate_terminal_envelope(
                    result, op_id=identity["op_id"], engine_id=identity["engine_id"],
                    engine_epoch=identity["engine_epoch"], owner=identity["owner"],
                    owner_seq=identity["owner_seq"], action=identity["action"],
                    slot=identity["slot"], filename=identity["filename"]):
                    raise SlotOperationError("startup persisted terminal envelope mismatch")
                metadata = json.loads(row["metadata"]) if row["metadata"] else None
                if metadata:
                    self._commit_index(metadata, result)
            elif row["state"] in ("published", "ack_pending"):
                raise SlotOperationError("startup persisted terminal result missing; backend remains unready")
            await self._ack(bounded_request, row["op_id"], row["seq"], row["epoch"], cap["owner"],
                            deadline=deadline)
        if publish:
            for metadata, result in self.published_entries():
                value = publish(metadata, result)
                if inspect.isawaitable(value):
                    await value
    def _index_key(self, metadata):
        kind = metadata.get("kind")
        if kind == "session":
            return kind, metadata.get("sid")
        if kind == "mode":
            return kind, f"{metadata.get('model')}:{metadata.get('slot')}"
        return None, None

    def _commit_index(self, metadata, result):
        kind, item_key = self._index_key(metadata)
        if not kind or not item_key:
            return
        with self._db() as db:
            db.execute("INSERT OR REPLACE INTO snapshot_index VALUES(?,?,?,?,?)",
                       (self.owner, kind, str(item_key), json.dumps(metadata, sort_keys=True),
                        json.dumps(result, sort_keys=True)))

    def is_quarantined_file(self, filename: str) -> bool:
        if not filename:
            return False
        with self._db() as db:
            row = db.execute(
                "SELECT 1 FROM operations WHERE filename=? AND state IN "
                "('pending','completed','failed','cancelled_before_start','published','ack_pending')",
                (filename,)
            ).fetchone()
            return row is not None

    def published_entries(self):
        with self._db() as db:
            rows = db.execute("SELECT metadata,result FROM snapshot_index WHERE owner=?", (self.owner,)).fetchall()
        return [(json.loads(row["metadata"]), json.loads(row["result"])) for row in rows]

    def remove_index(self, kind: str, item_key: str, *, expected_file: str | None = None) -> None:
        with self._db() as db:
            if expected_file is not None:
                row = db.execute("SELECT metadata FROM snapshot_index WHERE owner=? AND kind=? AND item_key=?",
                                 (self.owner, kind, str(item_key))).fetchone()
                if row:
                    try:
                        meta = json.loads(row["metadata"] if isinstance(row, sqlite3.Row) else row[0])
                        if meta.get("file") != expected_file:
                            return  # A newer replacement was committed; do not delete it!
                    except Exception:
                        pass
            db.execute("DELETE FROM snapshot_index WHERE owner=? AND kind=? AND item_key=?",
                       (self.owner, kind, str(item_key)))

    def _state(self, op_id, state, result):
        with self._db() as db:
            if state == "published" and result is not None:
                row = db.execute("SELECT owner,metadata FROM operations WHERE op_id=?", (op_id,)).fetchone()
                if row and row["metadata"]:
                    metadata = json.loads(row["metadata"])
                    kind, item_key = self._index_key(metadata)
                    if kind and item_key:
                        db.execute("INSERT OR REPLACE INTO snapshot_index VALUES(?,?,?,?,?)",
                                   (row["owner"], kind, str(item_key), json.dumps(metadata, sort_keys=True),
                                    json.dumps(result, sort_keys=True)))
            if state in ("acked", "stale_epoch"):
                if state == "acked":
                    row = db.execute("SELECT * FROM operations WHERE op_id=?", (op_id,)).fetchone()
                    identity = (persisted_operation_identity(
                        row, owner=self.owner, engine_id=self.engine_id,
                        engine_owner=self.engine_owner, endpoint=self.endpoint, model=self.model)
                        if row else None)
                    namespace_key = self._caller_namespace_key(identity) if identity else None
                    try:
                        terminal = json.loads(row["result"]) if row and row["result"] else None
                    except (TypeError, ValueError, json.JSONDecodeError):
                        terminal = None
                    refs = (db.execute(
                        "SELECT identity FROM caller_ack_hold_refs WHERE namespace_key=? AND op_id=?",
                        (namespace_key, op_id)).fetchall() if namespace_key else [])
                    if (identity and refs and all(ref["identity"] == json.dumps(identity, sort_keys=True)
                                                   for ref in refs)
                            and isinstance(terminal, dict)
                            and validate_terminal_envelope(
                                terminal, op_id=identity["op_id"], engine_id=identity["engine_id"],
                                engine_epoch=identity["engine_epoch"], owner=identity["owner"],
                                owner_seq=identity["owner_seq"], action=identity["action"],
                                slot=identity["slot"], filename=identity["filename"])):
                        encoded_identity = json.dumps(identity, sort_keys=True)
                        encoded_terminal = json.dumps(terminal, sort_keys=True)
                        existing = db.execute(
                            "SELECT identity,terminal FROM caller_ack_proofs_ns "
                            "WHERE namespace_key=? AND op_id=?", (namespace_key, op_id)).fetchone()
                        if not existing:
                            db.execute("INSERT INTO caller_ack_proofs_ns VALUES(?,?,?,?,?)",
                                       (namespace_key, op_id, encoded_identity,
                                        encoded_terminal, time.time()))
                        elif (existing["identity"] != encoded_identity
                              or existing["terminal"] != encoded_terminal):
                            # Conflicting prior evidence is preserved and never overwritten.
                            pass
                db.execute("DELETE FROM operations WHERE op_id=?", (op_id,))
                return
            db.execute("UPDATE operations SET state=?,result=?,updated=? WHERE op_id=?",
                       (state, json.dumps(result, sort_keys=True) if result is not None else None,
                        time.time(), op_id))
