from __future__ import annotations

from collections import defaultdict

BUCKETS = (0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, float("inf"))


def _lbl(**kw) -> str:
    return "{" + ",".join(f'{k}="{v}"' for k, v in kw.items()) + "}"


class Metrics:
    def __init__(self) -> None:
        self.requests: dict[tuple[str, str], int] = defaultdict(int)
        self.decisions: dict[tuple[str, str], int] = defaultdict(int)
        self.lat_sum: dict[str, float] = defaultdict(float)
        self.lat_count: dict[str, int] = defaultdict(int)
        self.lat_bucket: dict[tuple[str, float], int] = defaultdict(int)
        self.probe_sum: dict[str, float] = defaultdict(float)
        self.probe_count: dict[str, int] = defaultdict(int)
        self.probe_err: dict[tuple[str, str], int] = defaultdict(int)
        self.reused_est: dict[str, int] = defaultdict(int)      # tokens previstos ao rotear
        self.reused_obs: dict[str, int] = defaultdict(int)      # timings.cache_n observado
        self.aff_route: dict[str, int] = defaultdict(int)       # owner | skipped | cold
        self.aff_result: dict[tuple[str, str], int] = defaultdict(int)  # hit | miss por backend
        self.modes: dict[str, tuple[str, bool, bool]] = {}       # backend -> (modo, trocando, quebrado)
        self.switch_total: dict[tuple[str, str, str, str], int] = defaultdict(int)
        self.switch_sum: dict[str, float] = defaultdict(float)
        self.switch_count: dict[str, int] = defaultdict(int)
        self.switch_fail: dict[tuple[str, str], int] = defaultdict(int)
        self.slot_ops: dict[tuple[str, str, str], int] = defaultdict(int)
        self.handoffs: dict[str, int] = defaultdict(int)
        self.stage_sum: dict[str, float] = defaultdict(float)
        self.stage_count: dict[str, int] = defaultdict(int)
        self.http_requests: dict[tuple[str, str], int] = defaultdict(int)
        self.kv_refusals: dict[str, int] = defaultdict(int)
        self.duration_sum: dict[str, float] = defaultdict(float)
        self.duration_count: dict[str, int] = defaultdict(int)
        self.duration_bucket: dict[tuple[str, float], int] = defaultdict(int)
        self.request_n: dict[tuple[str, str], int] = defaultdict(int)
        self.schema_events: dict[tuple[str, str], int] = defaultdict(int)   # (acao, motivo)
        self.metrics_errors: int = 0

    def _safe(self, fn, *args) -> None:
        try:
            fn(*args)
        except Exception:
            try:
                self.metrics_errors += 1
            except Exception:
                pass

    def http_request(self, route: str, status: str) -> None:
        self.http_requests[(route, status)] += 1

    def kv_refusal(self, route: str) -> None:
        self.kv_refusals[route] += 1

    def schema_event(self, action: str, reason: str, n: int = 1) -> None:
        """Reparo de JSON Schema antes do forward. Só contagem: nunca o texto do pattern."""
        self.schema_events[(action, reason)] += n

    def duration(self, route: str, seconds: float) -> None:
        self.duration_sum[route] += seconds
        self.duration_count[route] += 1
        for b in BUCKETS:
            if seconds <= b:
                self.duration_bucket[(route, b)] += 1

    def request_n_observe(self, route: str, value) -> None:
        bucket = "missing" if value is None else ("n=1" if type(value) is int and value == 1 else
                                                    "n>1" if type(value) is int and value > 1 else None)
        if bucket is not None:
            self.request_n[(route, bucket)] += 1

    def handoff_stages(self, stages: dict[str, float]) -> None:
        for k, v in stages.items():
            self.stage_sum[k] += v
            self.stage_count[k] += 1

    def set_mode(self, backend: str, mode: str, switching: bool, broken: bool) -> None:
        self.modes[backend] = (mode, switching, broken)

    def mode_switch(self, backend: str, frm: str, to: str, result: str, seconds: float, step: str = "") -> None:
        self.switch_total[(backend, frm, to, result)] += 1
        if result in ("ok", "failed"):
            self.switch_sum[backend] += seconds
            self.switch_count[backend] += 1
        if step:
            self.switch_fail[(backend, step)] += 1

    def slot_op(self, backend: str, op: str, result: str) -> None:
        self.slot_ops[(backend, op, result)] += 1

    def decision(self, backend: str, reason: str) -> None:
        self.decisions[(backend, reason)] += 1

    def probe(self, stage: str, seconds: float, backend: str = "", err: bool = False) -> None:
        self.probe_sum[stage] += seconds
        self.probe_count[stage] += 1
        if err:
            self.probe_err[(stage, backend)] += 1

    def affinity_route(self, outcome: str, backend: str, predicted: int) -> None:
        self.aff_route[outcome] += 1
        if predicted > 0:
            self.reused_est[backend] += predicted

    def affinity_result(self, backend: str, predicted: int, observed: int | None) -> None:
        """Acerto = servidor reaproveitou >= 50% do previsto (cache_n do timings da resposta)."""
        if observed is None or predicted <= 0:
            return
        self.reused_obs[backend] += observed
        self.aff_result[(backend, "hit" if observed >= 0.5 * predicted else "miss")] += 1

    def request(self, backend: str, status: str, seconds: float) -> None:
        self.requests[(backend, status)] += 1
        self.lat_sum[backend] += seconds
        self.lat_count[backend] += 1
        for b in BUCKETS:
            if seconds <= b:
                self.lat_bucket[(backend, b)] += 1

    def render(self, states) -> str:
        o: list[str] = []
        o.append("# TYPE pool_http_requests_total counter")
        for (route, status), value in sorted(self.http_requests.items()):
            o.append(f"pool_http_requests_total{_lbl(route=route, status=status)} {value}")
        o.append("# HELP pool_kv_admission_refusals_total Upstream refusal events; retries may produce multiple events per HTTP request.")
        o.append("# TYPE pool_kv_admission_refusals_total counter")
        for route, value in sorted(self.kv_refusals.items()):
            o.append(f"pool_kv_admission_refusals_total{_lbl(route=route)} {value}")
        o.append("# TYPE pool_request_duration_seconds histogram")
        for route in sorted(self.duration_count):
            for b in BUCKETS:
                le = "+Inf" if b == float("inf") else str(b)
                o.append(f"pool_request_duration_seconds_bucket{_lbl(route=route, le=le)} {self.duration_bucket[(route, b)]}")
            o.append(f"pool_request_duration_seconds_sum{_lbl(route=route)} {self.duration_sum[route]:.6f}")
            o.append(f"pool_request_duration_seconds_count{_lbl(route=route)} {self.duration_count[route]}")
        o.append("# TYPE pool_schema_normalizations_total counter")
        for (action, reason), value in sorted(self.schema_events.items()):
            o.append(f"pool_schema_normalizations_total{_lbl(action=action, reason=reason)} {value}")
        o.append("# TYPE pool_request_n_total counter")
        for (route, bucket), value in sorted(self.request_n.items()):
            o.append(f"pool_request_n_total{_lbl(route=route, n_bucket=bucket)} {value}")
        o.append(f"# TYPE pool_metrics_errors_total counter")
        o.append(f"pool_metrics_errors_total {self.metrics_errors}")
        o.append("# TYPE pool_handoff_total counter")
        for result, count in sorted(self.handoffs.items()):
            o.append(f"pool_handoff_total{_lbl(result=result)} {count}")
        o.append("# TYPE pool_handoff_stage_seconds summary")
        for k in sorted(self.stage_count):
            o.append(f"pool_handoff_stage_seconds_sum{_lbl(stage=k)} {self.stage_sum[k]:.6f}")
            o.append(f"pool_handoff_stage_seconds_count{_lbl(stage=k)} {self.stage_count[k]}")
        o.append("# TYPE pool_requests_total counter")
        for (b, s), v in sorted(self.requests.items()):
            o.append(f"pool_requests_total{_lbl(backend=b, status=s)} {v}")
        o.append("# TYPE pool_route_decisions_total counter")
        for (b, r), v in sorted(self.decisions.items()):
            o.append(f"pool_route_decisions_total{_lbl(backend=b, reason=r)} {v}")
        o.append("# TYPE pool_request_seconds histogram")
        for b in sorted(self.lat_count):
            for le in BUCKETS:
                les = "+Inf" if le == float("inf") else str(le)
                o.append(f"pool_request_seconds_bucket{_lbl(backend=b, le=les)} {self.lat_bucket[(b, le)]}")
            o.append(f"pool_request_seconds_sum{_lbl(backend=b)} {self.lat_sum[b]:.6f}")
            o.append(f"pool_request_seconds_count{_lbl(backend=b)} {self.lat_count[b]}")
        o.append("# TYPE pool_prefix_reused_tokens_estimated_total counter")
        for b, v in sorted(self.reused_est.items()):
            o.append(f"pool_prefix_reused_tokens_estimated_total{_lbl(backend=b)} {v}")
        o.append("# TYPE pool_prefix_reused_tokens_observed_total counter")
        for b, v in sorted(self.reused_obs.items()):
            o.append(f"pool_prefix_reused_tokens_observed_total{_lbl(backend=b)} {v}")
        o.append("# TYPE pool_affinity_route_total counter")
        for k, v in sorted(self.aff_route.items()):
            o.append(f"pool_affinity_route_total{_lbl(outcome=k)} {v}")
        o.append("# TYPE pool_affinity_result_total counter")
        for (b, k), v in sorted(self.aff_result.items()):
            o.append(f"pool_affinity_result_total{_lbl(backend=b, result=k)} {v}")
        o.append("# TYPE pool_probe_seconds summary")
        for st_ in sorted(self.probe_count):
            o.append(f"pool_probe_seconds_sum{_lbl(stage=st_)} {self.probe_sum[st_]:.6f}")
            o.append(f"pool_probe_seconds_count{_lbl(stage=st_)} {self.probe_count[st_]}")
        o.append("# TYPE pool_probe_errors_total counter")
        for (st_, b), v in sorted(self.probe_err.items()):
            o.append(f"pool_probe_errors_total{_lbl(stage=st_, backend=b)} {v}")
        o.append("# TYPE pool_queue_depth gauge")
        for st in states:
            o.append(f"pool_queue_depth{_lbl(backend=st.cfg.name)} {st.queued}")
        o.append("# TYPE pool_inflight gauge")
        for st in states:
            o.append(f"pool_inflight{_lbl(backend=st.cfg.name)} {st.inflight}")
        o.append("# TYPE pool_backend_up gauge")
        for st in states:
            o.append(f"pool_backend_up{_lbl(backend=st.cfg.name)} {1 if st.healthy else 0}")
        o.append("# TYPE pool_service_seconds_avg gauge")
        for st in states:
            o.append(f"pool_service_seconds_avg{_lbl(backend=st.cfg.name)} {st.avg_s:.3f}")
        if self.modes or self.switch_total:
            o.append("# TYPE pool_mode_switch_total counter")
            for (b, f, t, r), v in sorted(self.switch_total.items()):
                o.append(f"pool_mode_switch_total{_lbl(backend=b, **{'from': f}, to=t, result=r)} {v}")
            o.append("# TYPE pool_mode_switch_seconds summary")
            for b in sorted(self.switch_count):
                o.append(f"pool_mode_switch_seconds_sum{_lbl(backend=b)} {self.switch_sum[b]:.6f}")
                o.append(f"pool_mode_switch_seconds_count{_lbl(backend=b)} {self.switch_count[b]}")
            o.append("# TYPE pool_mode_switch_failures_total counter")
            for (b, step), v in sorted(self.switch_fail.items()):
                o.append(f"pool_mode_switch_failures_total{_lbl(backend=b, step=step)} {v}")
            o.append("# TYPE pool_mode_slot_ops_total counter")
            for (b, op, r), v in sorted(self.slot_ops.items()):
                o.append(f"pool_mode_slot_ops_total{_lbl(backend=b, op=op, result=r)} {v}")
            o.append("# TYPE pool_backend_mode gauge")
            for b, (m, _sw, _br) in sorted(self.modes.items()):
                o.append(f"pool_backend_mode{_lbl(backend=b, mode=m)} 1")
            o.append("# TYPE pool_mode_switching gauge")
            for b, (_m, sw, _br) in sorted(self.modes.items()):
                o.append(f"pool_mode_switching{_lbl(backend=b)} {1 if sw else 0}")
        return "\n".join(o) + "\n"
