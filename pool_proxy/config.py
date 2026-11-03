from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

import yaml

from .paths import state_path


@dataclass
class ModePolicy:
    """Politica de troca de modo de um backend em router mode (tempos em segundos)."""
    demand_window_s: float = 300.0       # janela deslizante de demanda por modelo pedido
    switch_min_dwell_s: float = 60.0     # demanda do outro modo tem de se estender por este intervalo
    switch_min_requests: int = 1         # minimo de pedidos do outro modo na janela
    min_mode_s: float = 600.0            # tempo minimo no modo atual
    switch_margin_requests: int = 0      # >0: com o modo atual em uso (pedido em voo), a demanda esperando deve ser >= demanda dele + margem (1 = empate nao troca)
    busy_threshold_requests: int = 2     # modo atual com menos pedidos que isto na janela conta como "abaixo do limiar"
    drain_timeout_s: float = 120.0       # espera pelos pedidos em voo antes de abortar a troca
    switch_timeout_s: float = 120.0      # espera maxima de um pedido por modelo indisponivel
    load_timeout_s: float = 300.0
    unload_timeout_s: float = 180.0      # MI50: stop-timeout do preset = 120
    op_timeout_s: float = 60.0           # timeout HTTP das chamadas de gerencia e de slot save/restore
    poll_s: float = 0.5
    eval_interval_s: float = 5.0
    fail_backoff_s: float = 30.0         # pausa apos troca que falhou/abortou
    recover_interval_s: float = 30.0     # tentativas de recuperar backend marcado unhealthy
    save_slots: bool = True
    save_min_tokens: int = 256           # so salva slot com cache >= isto
    swap_estimate_s: float = 30.0        # penalidade inicial de custo para modelo nao carregado
    alive_check_s: float = 15.0          # confere no router se os modelos do modo atual seguem rodando (0 = nunca)
    low_switch: bool = False             # demanda da faixa low pode trocar o modo (so com o router sem uso de primeiro plano)
    low_switch_dwell_s: float = 60.0     # a demanda low tem de esperar pelo menos isto antes de trocar
    low_switch_min_requests: int = 1     # minimo de pedidos low esperando pelo outro modo


@dataclass
class BackendCfg:
    name: str
    url: str
    engine_id: str = ""  # stable configured ID for a direct engine endpoint
    engine_owner: str = ""  # expected authenticated proxy principal at a direct endpoint
    model_engine_ids: dict[str, str] = field(default_factory=dict)  # exact child IDs in router mode
    model_engine_owners: dict[str, str] = field(default_factory=dict)  # authenticated proxy principal per child
    slotops_ledger_dir: str = field(default_factory=lambda: state_path("slot-operations"))
    key_env: str = ""          # nome da variavel de ambiente com a chave (nunca a chave)
    vision: bool = False
    ctx_max: int = 0           # 0 = sem limite conhecido
    kv_unified_ctx: int = 0    # >0: KV unificado compartilhado pelos slots (tokens); a faixa low libera slots para caber
    np: int = 1                # slots = concorrencia maxima
    weight: float = 1.0        # >1 favorece o backend no custo
    prefill_tps: float = 500.0
    decode_tps: float = 30.0
    overflow: bool = False     # destino de spill; fora do roteamento normal
    models: list[str] = field(default_factory=list)  # backend direto: modelos do cliente que ele serve (vazio = todos)
    router: bool = False       # llama-server em router mode (varios modelos, um arranjo ativo por vez)
    modes: dict[str, list[str]] = field(default_factory=dict)   # modo -> modelos carregados nele
    default_mode: str = ""
    default_model: str = ""    # modelo usado quando o pedido nao nomeia um modelo conhecido do router
    model_aliases: dict[str, str] = field(default_factory=dict)  # alias do cliente -> nome canonico
    no_slot_save: list[str] = field(default_factory=list)  # modelos sem --slot-save-path: nao salvar/restaurar slots
    mode_policy: ModePolicy = field(default_factory=ModePolicy)
    handoff_model_path: str = ""  # caminho exato anunciado em /props
    handoff_gguf_path: str = ""   # mesmo GGUF, montado read-only no proxy para SHA256
    handoff_state_format: str = ""  # contrato verificado: build/ABI, K/V, rotacao e patches
    handoff_model: str = ""         # destino router: modelo canonico (secao do preset) que recebe o handoff
    handoff_preset: dict[str, str] = field(default_factory=dict)  # destino router: opcoes que o preset do modelo deve declarar (cache-type-k, cache-type-v, kv-unified...)


@dataclass
class Handoff35:
    """Handoff do 35B: prefill na 3090 (engine de troca) -> decode no mi50-35b."""
    enabled: bool = False
    # aliases do cliente que disparam o handoff (o gate exige o modelo 35B)
    models: list[str] = field(default_factory=lambda: ["qwen3.6-35b", "batch/35b"])
    router_model: str = "mi50-35b"       # nome canonico do destino no router
    min_tokens: int = 64000              # gate: prompt estimado/exato
    max_cached_tokens: int = 8192        # cache util no MI50: acima disto nao vale a troca
    source: str = "rtx3090-27b"          # backend que cede a placa (parado com dreno)
    target: str = "mi50-router"          # backend que decodifica
    ctl_url: str = "http://rtx3090-host:8017"    # prefill35-ctl
    ctl_key_env: str = "POOL_KEY_PREFILL35"
    ctl_timeout_s: float = 120.0
    open_timeout_s: float = 300.0        # stop 27B + load 35B (~13 s medidos; folga p/ VRAM fria)
    close_timeout_s: float = 300.0       # stop 35B + volta do 27B + prova POST
    lease_timeout_s: float = 150.0       # troca de modo m27->m35 no router
    reconcile_timeout_s: float = 20.0
    tokens_timeout_s: float = 120.0
    stage_timeout_s: float = 180.0       # chamadas de slot/capacidade
    open_ttl_s: float = 900.0            # TTL da janela no ctl (sem heartbeat => 27B volta)
    hold_wait_s: float = 90.0            # fila curta dos pedidos 27B durante a janela
    health_interval_s: float = 30.0      # /state do ctl (janela orfa, readiness)
    engine_id: str = "prefill35-3090"    # --slot-operation-engine-id do engine de prefill
    engine_owner: str = "proxy"
    source_np: int = 2
    target_np: int = 2
    source_ctx_slot: int = 131072        # ctx/slot do engine de prefill (-c 262144 -np 2)
    expected_preset: dict[str, str] = field(
        default_factory=lambda: {"cache-type-k": "q8_0", "cache-type-v": "q8_0"})
    prefills_tps: float = 1800.0         # prefill medido do 35B na 3090 (1797-2426 t/s)
    prefill_margin: float = 1.6
    stage_overhead_s: float = 90.0
    ledger_dir: str = field(default_factory=lambda: state_path("slot-operations"))

    def prefill_budget_s(self, tokens: int) -> float:
        if tokens <= 0 or self.prefills_tps <= 0:
            return self.stage_timeout_s
        return max(self.stage_timeout_s,
                   tokens / self.prefills_tps * self.prefill_margin + self.stage_overhead_s)


@dataclass
class Settings:
    affinity_max_wait_s: float = 30.0   # afinidade so se a espera estimada couber
    spill_after_s: float = 60.0         # espera estimada acima disto -> overflow
    queue_timeout_s: float = 120.0      # espera real maxima por slot
    long_prompt_tokens: int = 3000      # prompt frio acima disto -> maior prefill_tps
    chars_per_token: float = 3.5
    image_tokens: int = 1000
    default_max_tokens: int = 512
    default_service_s: float = 10.0     # tempo medio inicial por requisicao
    ema_alpha: float = 0.2
    health_interval_s: float = 10.0
    health_timeout_s: float = 5.0
    health_failures: int = 2            # falhas consecutivas para marcar down
    connect_timeout_s: float = 5.0
    max_attempts: int = 3
    affinity_capacity: int = 50000
    probe_enabled: bool = True          # SlotProbe slot-telemetry (sem slot-telemetry cai no espelho local)
    probe_slots_timeout_s: float = 0.8  # /slots passa pela fila do servidor: timeout baixo
    probe_slots_ttl_s: float = 1.5
    probe_tokenize_timeout_s: float = 3.0
    probe_cache_size: int = 256         # LRU de tokenizacao por hash do corpo
    probe_recheck_s: float = 300.0      # reteste de backend marcado sem slot-telemetry
    low_priority_header: str = "x-pool-priority"   # valor "low" -> faixa de baixa prioridade
    low_idle_s: float = 15.0            # sem pedido de PRIMEIRO PLANO ha pelo menos isto para aceitar um pedido low
    low_max_inflight: int = 1           # pedidos low simultaneos por backend (limitado a np); low seguido de low nao espera
    low_sticky_tokens: int = 8192       # dono do cache com pelo menos isto reaproveitavel: esperar por ele
    low_sticky_wait_s: float = 30.0     # ... ate isto; depois a migracao de KV (sessao) cobre a troca de backend
    low_session: bool = False           # KV de sessao dos workers low persiste entre turnos (sessions.py)
    low_session_min_tokens: int = 8192  # so salva/restaura estados com pelo menos isto
    low_session_max_bytes: int = 6 * 1024**3   # teto do uso REAL dos sess-* (inclui .ckpt) no tmpfs; despejo LRU
    low_session_save_every_tokens: int = 0   # >0: so salva quando o arquivo da sessao esta >= N tokens atras; o slot
                                             # fica "sujo" e e salvo antes de ser sobrescrito (outra sessao, 1o plano)
    low_handoff: bool = False            # faixa low: prefill do que falta na origem do handoff (3090), decode no destino
    low_handoff_min_tokens: int = 8192
    low_handoff_est_factor: float = 1.0  # estimativa sem tokenizar sai 30-40% abaixo do real em contexto de agente
    low_handoff_wait_s: float = 0.0       # espera a origem ociosa (turnos low na 3090 duram segundos; prefill frio nas MI50, minutos)   # tokens a prefillar no destino (estimativa - estado restauravel) para valer
    low_session_min_free_bytes: int = 2 * 1024**3  # folga minima no tmpfs antes de salvar (handoff usa o mesmo)
    low_session_save_from: list[str] = field(default_factory=list)  # backends cujo save e seguro (MI50 split tensor: assert t43)
    low_wait_s: float = 1800.0          # espera maxima de um pedido low por um vao (inclui janelas de GPU)
    low_poll_s: float = 2.0
    # reparo dos JSON Schemas de tools/response_format antes do forward (caminho unico).
    # `schema_normalize: false` desliga tudo (rollback de 1 linha, sem trocar codigo).
    schema_normalize: bool = True
    schema_minmax_cap: int = 1000       # teto medido: maxItems 1000 ok, 2000 = 400 do conversor
    client_key_env: str = ""          # variavel com a chave exigida dos clientes (vazio = sem auth)
    handoff_enabled: bool = False
    handoff_source: str = ""
    handoff_target: str = ""
    handoff_models: list[str] = field(default_factory=list)
    handoff_min_tokens: int = 8192
    handoff_timeout_s: float = 180.0
    handoff_prefill_margin: float = 1.6  # orcamento do prefill da origem = tokens/prefill_tps * isto + folga;
                                         # o timeout NAO pode ser menor que o prefill real do prompt
    handoff_stage_overhead_s: float = 60.0  # folga no mesmo orcamento para template/lease/transfer/restore
    handoff_quiesce_s: float = 5.0      # apos falha: espera o slot da origem sair de is_processing (reuso LCP do caminho normal)
    handoff_transport: str = "shared"  # shared ou copy (diretorios montados no proxy)
    handoff_source_dir: str = ""
    handoff_target_dir: str = ""
    handoff_switch_overlap: bool = True  # destino router: troca/carga do modelo durante o prefill da origem
    handoff_35b: Handoff35 = field(default_factory=Handoff35)

    def handoff_prefill_budget_s(self, tokens: int, prefill_tps: float) -> float:
        """Orcamento de tempo do prefill+save da origem do handoff.

        handoff_timeout_s continua sendo o limite POR chamada HTTP (e o piso deste orcamento);
        o prefill de um prompt grande (~130k tokens) nao cabe nele: 130k a 629 t/s = 207 s e a
        240 s o orcamento ja era 86% consumido. O orcamento cresce com o numero
        EXATO de tokens e com a taxa de prefill medida do backend, com margem configuravel.
        """
        if tokens <= 0 or prefill_tps <= 0:
            return self.handoff_timeout_s
        need = tokens / prefill_tps * self.handoff_prefill_margin + self.handoff_stage_overhead_s
        return max(self.handoff_timeout_s, need)


@dataclass
class Config:
    backends: list[BackendCfg]
    settings: Settings = field(default_factory=Settings)
    models: list[str] = field(default_factory=lambda: ["pool"])
    host: str = "0.0.0.0"
    port: int = 8100


def _build(cls, data: dict[str, Any]):
    allowed = {f.name for f in fields(cls)}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"{cls.__name__}: chaves desconhecidas {sorted(unknown)}")
    return cls(**data)


def _backend(d: dict[str, Any]) -> BackendCfg:
    d = dict(d)
    pol = d.pop("mode_policy", None) or {}
    b = _build(BackendCfg, d)
    b.mode_policy = _build(ModePolicy, pol)
    if b.router:
        if not b.modes or any(not v for v in b.modes.values()):
            raise ValueError(f"backend {b.name}: router exige `modes` nao vazios")
        b.default_mode = b.default_mode or next(iter(b.modes))
        if b.default_mode not in b.modes:
            raise ValueError(f"backend {b.name}: default_mode desconhecido")
        known = {m for v in b.modes.values() for m in v}
        if b.default_model and b.default_model not in known:
            raise ValueError(f"backend {b.name}: default_model fora dos modos")
        if set(b.model_engine_ids) - known:
            raise ValueError(f"backend {b.name}: model_engine_ids contém modelo fora dos modos")
        for m in b.no_slot_save:
            if m not in known:
                raise ValueError(f"backend {b.name}: no_slot_save {m} fora dos modos")
        for a, m in b.model_aliases.items():
            if m not in known:
                raise ValueError(f"backend {b.name}: alias {a} aponta para modelo fora dos modos")
    if b.router:
        models = {m for group in b.modes.values() for m in group}
        if any(not b.model_engine_ids.get(m) or not b.model_engine_owners.get(m) for m in models):
            raise ValueError(f"backend {b.name}: cada modelo router exige model_engine_ids e model_engine_owners estáveis")
        configured_ids = [b.model_engine_ids[m] for m in models]
        if len(set(configured_ids)) != len(configured_ids):
            raise ValueError(f"backend {b.name}: model_engine_ids deve ser único por filho")
        if set(b.model_engine_owners) - known:
            raise ValueError(f"backend {b.name}: model_engine_owners contém modelo fora dos modos")
    return b


def from_dict(d: dict[str, Any]) -> Config:
    backends = [_backend(b) for b in d.get("backends", [])]
    if not backends:
        raise ValueError("config sem backends")
    names = [b.name for b in backends]
    if len(set(names)) != len(names):
        raise ValueError("nomes de backend duplicados")
    raw_settings = dict(d.get("settings") or {})
    raw_h35 = raw_settings.pop("handoff_35b", None) or {}
    settings = _build(Settings, raw_settings)
    if raw_h35:
        settings.handoff_35b = _build(Handoff35, raw_h35)
    h35 = settings.handoff_35b
    if h35.enabled:
        src = next((b for b in backends if b.name == h35.source), None)
        dst = next((b for b in backends if b.name == h35.target), None)
        if src is None or dst is None or src is dst:
            raise ValueError("handoff_35b: source/target devem ser backends distintos e existentes")
        if src.router or not dst.router or src.overflow or dst.overflow:
            raise ValueError("handoff_35b: origem direta (sem overflow), destino router")
        known = {m for v in dst.modes.values() for m in v}
        if h35.router_model not in known:
            raise ValueError("handoff_35b: router_model fora dos modos do destino")
        if not h35.models or h35.min_tokens <= 0:
            raise ValueError("handoff_35b: models e min_tokens obrigatorios")
        if not h35.ctl_url:
            raise ValueError("handoff_35b: ctl_url obrigatorio")
    if settings.handoff_enabled:
        pair = [next((b for b in backends if b.name == n), None)
                for n in (settings.handoff_source, settings.handoff_target)]
        if any(b is None for b in pair) or pair[0] is pair[1]:
            raise ValueError("handoff: source/target devem ser backends distintos")
        if any(b.overflow for b in pair) or pair[0].router:
            raise ValueError("handoff: origem direta; destino direto ou router (sem overflow)")
        if pair[1].router:
            known = {m for v in pair[1].modes.values() for m in v}
            if pair[1].handoff_model not in known:
                raise ValueError("handoff: destino router exige handoff_model entre os modelos dos modos")
            if pair[1].handoff_model in pair[1].no_slot_save:
                raise ValueError("handoff: handoff_model nao pode estar em no_slot_save")
        if any(b.np < 1 for b in pair):
            raise ValueError("handoff: np deve ser positivo")
        if not settings.handoff_models or settings.handoff_min_tokens <= 0 or settings.handoff_timeout_s <= 0:
            raise ValueError("handoff: modelos, limiar e timeout obrigatorios")
        if settings.handoff_transport not in ("shared", "copy"):
            raise ValueError("handoff: transporte deve ser shared ou copy")
        if not settings.handoff_source_dir or not settings.handoff_target_dir:
            raise ValueError("handoff: diretorios de slots obrigatorios")
        if any(not b.handoff_model_path or not b.handoff_gguf_path or not b.handoff_state_format for b in pair):
            raise ValueError("handoff: identidade GGUF e contrato de estado obrigatorios")
        if pair[0].handoff_state_format != pair[1].handoff_state_format:
            raise ValueError("handoff: contratos de estado diferentes")
    return Config(
        backends=backends,
        settings=settings,
        models=list(d.get("models", ["pool"])),
        host=d.get("host", "0.0.0.0"),
        port=int(d.get("port", 8100)),
    )


def load(path: str) -> Config:
    with open(path, encoding="utf-8") as f:
        return from_dict(yaml.safe_load(f) or {})
