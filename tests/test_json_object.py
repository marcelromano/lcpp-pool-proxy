import json

from pool_proxy.app import _json_object_to_schema


def test_json_object_vira_schema():
    body = {"model": "m", "response_format": {"type": "json_object"}}
    raw, out = _json_object_to_schema(json.dumps(body).encode(), body)
    rf = json.loads(raw)["response_format"]
    assert rf == out["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["schema"] == {"type": "object"}


def test_outros_formatos_intactos():
    for rf in (None, {"type": "text"}, {"type": "json_schema", "json_schema": {"schema": {}}},
               {"type": "json_object", "schema": {"type": "array"}}):
        body = {"model": "m"} if rf is None else {"model": "m", "response_format": rf}
        raw = json.dumps(body).encode()
        assert _json_object_to_schema(raw, body) == (raw, body)
