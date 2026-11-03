"""Cliente da API de gerencia do llama-server em router mode (mainline). See README (Architecture)."""
from __future__ import annotations

import asyncio
from contextlib import closing
import json
import time
import hashlib
import sqlite3
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Iterable


import aiohttp
from .paths import state_path
from .slotops import SlotOperations

RUNNING = ("loaded", "loading", "sleeping")


class RouterError(Exception):
    def __init__(self, message: str, status: int = 0) -> None:
        super().__init__(message)
        self.status = status


class ModelFailed(RouterError):
    """O filho do modelo saiu com erro durante o load (`status.failed`)."""


class WaitTimeout(RouterError):
    pass


@dataclass
class ModelState:
    name: str
    status: str = "unloaded"
    failed: bool = False
    exit_code: int | None = None
    progress: Any = None

    @property
    def running(self) -> bool:
        return self.status in RUNNING


def slot_filename(model: str, slot: int) -> str:
    """Nome deterministico (so [A-Za-z0-9._-], aceito por fs_validate_filename)."""
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in model)
    return f"pp-{safe}-slot{slot}.bin"


class RouterCtl:
    def __init__(self, session_getter: Callable[[], aiohttp.ClientSession], base_url: str,
                 headers_fn: Callable[[], dict[str, str]] | None = None, op_timeout_s: float = 60.0,
                 poll_s: float = 0.5, ledger_dir: str | Path | None = None,
                 engine_ids: dict[str, str] | None = None,
                 engine_owners: dict[str, str] | None = None,
                 ledger_namespace: str = "") -> None:
        self._session = session_getter
        self.base = base_url.rstrip("/")
        self._headers = headers_fn or (lambda: {})
        self.op_timeout_s = op_timeout_s
        self.poll_s = poll_s
        self.ledger_dir = Path(ledger_dir if ledger_dir is not None else state_path("slot-operations"))
        self.engine_ids = dict(engine_ids or {})
        self.engine_owners = dict(engine_owners or {})
        self._slot_operations = {}
        self.ledger_namespace = ledger_namespace or self.base
        self._ledger_scope = hashlib.sha256(self.ledger_namespace.encode()).hexdigest()[:20]

    def slot_operations_for(self, model: str) -> SlotOperations:
        client = self._slot_operations.get(model)
        if client is None:
            identity = self.base + "\0" + model
            model_key = hashlib.sha256(model.encode()).hexdigest()[:16]
            db = self.ledger_dir / f"pool-router-slotops-{self._ledger_scope}-{model_key}.sqlite3"
            client = self._slot_operations[model] = SlotOperations(
                db, "router:" + hashlib.sha256(identity.encode()).hexdigest(), timeout_s=self.op_timeout_s,
                engine_id=self.engine_ids.get(model, ""), engine_owner=self.engine_owners.get(model, ""),
                endpoint=self.base, model=model)
        return client

    def pending_operation_models(self) -> set[str]:
        """Discover every durable child ledger in this router scope before startup admission."""
        models = set()
        pattern = f"pool-router-slotops-{self._ledger_scope}-*.sqlite3"
        for path in self.ledger_dir.glob(pattern):
            try:
                with closing(sqlite3.connect(path)) as db:
                    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
                    rows = []
                    if "operations" in tables:
                        rows = db.execute("SELECT DISTINCT engine_id FROM operations WHERE state IN "
                                          "('pending','completed','failed','cancelled_before_start','published','ack_pending')").fetchall()
                    has_index = False
                    if "snapshot_index" in tables:
                        has_index = db.execute("SELECT COUNT(*) FROM snapshot_index").fetchone()[0] > 0
            except sqlite3.Error as exc:
                raise RouterError("router operation ledger unreadable; startup remains blocked") from exc
            for (engine_id,) in rows:
                if not isinstance(engine_id, str) or engine_id not in self.engine_ids.values():
                    raise RouterError("router operation ledger has unknown engine identity")
                model = next(model for model, configured_id in self.engine_ids.items()
                             if configured_id == engine_id)
                if not model:
                    raise RouterError("router operation ledger has empty model identity")
                models.add(model)
            if has_index:
                for model in self.engine_ids:
                    m_key = hashlib.sha256(model.encode()).hexdigest()[:16]
                    if path.name == f"pool-router-slotops-{self._ledger_scope}-{m_key}.sqlite3":
                        models.add(model)
        return models

    async def _req(self, method: str, path: str, *, params: dict | None = None, body: dict | None = None,
                   timeout: float | None = None) -> Any:
        try:
            async with self._session().request(
                    method, self.base + path, params=params, json=body, headers=self._headers(),
                    timeout=aiohttp.ClientTimeout(total=timeout or self.op_timeout_s)) as r:
                txt = await r.text()
                if r.status >= 300:
                    raise RouterError(f"{method} {path} -> {r.status}: {txt[:300]}", r.status)
                if not txt:
                    return {}
                try:
                    return json.loads(txt)
                except ValueError:
                    raise RouterError(f"{method} {path}: resposta nao e JSON", r.status) from None
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            raise RouterError(f"{method} {path}: {type(e).__name__}") from None

    # ---- estado ------------------------------------------------------------
    async def props(self) -> dict:
        """GET /props sem `model`: {role:"router", max_instances, models_autoload}."""
        return await self._req("GET", "/props")

    async def list_models(self) -> dict[str, ModelState]:
        """GET /models: data[].id, data[].status.{value,failed,exit_code,progress}."""
        data = (await self._req("GET", "/models")).get("data") or []
        out: dict[str, ModelState] = {}
        for m in data:
            st = m.get("status") or {}
            out[m["id"]] = ModelState(m["id"], st.get("value", "unloaded"), bool(st.get("failed")),
                                      st.get("exit_code"), st.get("progress"))
        return out

    # ---- carga -------------------------------------------------------------
    async def load(self, model: str) -> None:
        """POST /models/load {model}: assincrono; 400 se o modelo ja roda."""
        await self._req("POST", "/models/load", body={"model": model})

    async def unload(self, model: str) -> None:
        """POST /models/unload {model}: assincrono; 400 se nao roda."""
        await self._req("POST", "/models/unload", body={"model": model})

    async def wait_state(self, model: str, want: Iterable[str], timeout: float) -> ModelState:
        """Poll de GET /models ate `status.value` estar em `want`. `failed` durante espera de
        loaded levanta ModelFailed; estourar o prazo levanta WaitTimeout."""
        want = set(want)
        deadline = time.monotonic() + timeout
        while True:
            states = await self.list_models()
            ms = states.get(model)
            if ms is None:
                raise RouterError(f"modelo {model} desconhecido do router", 404)
            if ms.status in want:
                return ms
            if ms.failed and ms.status == "unloaded" and want & {"loaded", "sleeping"}:
                raise ModelFailed(f"modelo {model} falhou no load (exit_code={ms.exit_code})")
            if time.monotonic() >= deadline:
                raise WaitTimeout(f"modelo {model}: {ms.status} apos {timeout:.0f}s, esperado {sorted(want)}")
            await asyncio.sleep(self.poll_s)

    async def wait_loaded(self, model: str, timeout: float) -> ModelState:
        return await self.wait_state(model, ("loaded", "sleeping"), timeout)

    async def wait_unloaded(self, model: str, timeout: float) -> ModelState:
        return await self.wait_state(model, ("unloaded",), timeout)

    # ---- slots por modelo ----------------------------------------------------
    async def slots(self, model: str) -> list[dict]:
        """GET /slots?model=X&autoload=false. SEM autoload=false o router carrega X. Chamar so com
        o modelo ja carregado (descarregado devolve 400 'model is not loaded')."""
        r = await self._req("GET", "/slots", params={"model": model, "autoload": "false"})
        return r if isinstance(r, list) else []

    async def slot_save(self, model: str, slot: int, filename: str, *, publish=None, metadata=None) -> dict:
        """POST /slots/N?action=save com `model` no corpo (o router so roteia POST por body.model)."""
        return await self._slot_action("save", model, slot, filename, publish=publish, metadata=metadata)

    async def slot_restore(self, model: str, slot: int, filename: str) -> dict:
        return await self._slot_action("restore", model, slot, filename)

    async def _slot_action(self, action: str, model: str, slot: int, filename: str,
                           *, publish=None, metadata=None) -> dict:
        operation_request = self.slotop_request(model)
        return await self.slot_operations_for(model).execute(operation_request, action, int(slot), filename,
                                                  {"model": model}, publish=publish, metadata=metadata)

    def recover_published_operation(self, metadata, envelope):
        """ModeManager attaches its index callback during startup reconciliation."""
        if self._publish_recovery:
            self._publish_recovery(metadata, envelope)

    _publish_recovery = None

    def slotop_request(self, model: str):
        """Transport callback that preserves router model routing and disables autoload."""
        async def operation_request(method, path, body):
            params = {"autoload": "false"}
            if method == "POST":
                body = {**(body or {}), "model": model}
            else:
                params["model"] = model
            return await self._req(method, path, params=params, body=body)
        return operation_request


