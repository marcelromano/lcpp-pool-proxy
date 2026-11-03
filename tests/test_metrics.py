from pool_proxy.metrics import BUCKETS, Metrics


def test_metric_names_and_route_labels():
    m = Metrics()
    m.http_request("/v1/chat/completions", "200")
    text = m.render([])
    assert 'pool_http_requests_total{route="/v1/chat/completions",status="200"} 1' in text


def test_kv_refusal_counter():
    m = Metrics()
    m.kv_refusal("/v1/chat/completions")
    assert 'pool_kv_admission_refusals_total{route="/v1/chat/completions"} 1' in m.render([])


def test_histogram_boundaries_include_requested_20_second_bucket():
    assert BUCKETS == (0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, float("inf"))
    m = Metrics()
    m.duration("/v1/chat/completions", 20)
    text = m.render([])
    assert 'le="20"} 1' in text and 'le="+Inf"} 1' in text


def test_n_is_classified_without_literal_cardinality():
    m = Metrics()
    for value in (1, 2, 1000000):
        m.request_n_observe("/v1/chat/completions", value)
    text = m.render([])
    assert 'n_bucket="n=1"} 1' in text
    assert 'n_bucket="n>1"} 2' in text
    assert "1000000" not in text


def test_missing_n_bucket():
    m = Metrics()
    m.request_n_observe("/v1/chat/completions", None)
    assert 'n_bucket="missing"} 1' in m.render([])


def test_bad_n_does_not_create_arbitrary_label():
    m = Metrics()
    for value in (True, 0, -1, 1.5, "2"):
        m.request_n_observe("/v1/chat/completions", value)
    assert "pool_request_n_total{route" not in m.render([])


def test_counter_is_monotonic_and_histogram_counts_once():
    m = Metrics()
    m.http_request("/v1/chat/completions", "503")
    m.http_request("/v1/chat/completions", "503")
    m.duration("/v1/chat/completions", 0.25)
    text = m.render([])
    assert 'status="503"} 2' in text
    assert 'pool_request_duration_seconds_count{route="/v1/chat/completions"} 1' in text


def test_metrics_failure_is_counted_and_not_propagated():
    m = Metrics()
    class Broken:
        def fail(self):
            raise RuntimeError("injected")
    m._safe(Broken().fail)
    assert "pool_metrics_errors_total 1" in m.render([])
