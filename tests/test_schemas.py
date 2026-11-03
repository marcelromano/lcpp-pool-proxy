""" — reparo dos JSON Schemas de ferramentas do caminho unico (pool-proxy)."""

import json

from conftest import Fake
from pool_proxy import schemas

U = [{"role": "user", "content": "oi"}]


def _body(pattern, where="items", model="pool"):
    if where == "items":
        params = {"type": "object", "required": ["a"],
                  "properties": {"a": {"type": "array", "items": {"type": "string",
                                                                 "pattern": pattern}}}}
    elif where == "prop":
        params = {"type": "object", "properties": {"u": {"type": "string", "pattern": pattern}},
                  "required": ["u"]}
    elif where == "nested":
        params = {"type": "object", "required": ["o"],
                  "properties": {"o": {"type": "object", "required": ["i"],
                                       "properties": {"i": {"type": "string", "pattern": pattern}}}}}
    else:
        raise AssertionError(where)
    return {"model": model, "messages": U, "tool_choice": "auto",
            "tools": [{"type": "function", "function": {"name": "z", "description": "d",
                                                        "parameters": params}}]}


def _pat(body):
    return body["tools"][0]["function"]["parameters"]["properties"][
        "a" if "a" in body["tools"][0]["function"]["parameters"]["properties"] else "u"]["pattern"] \
        if "a" in body["tools"][0]["function"]["parameters"]["properties"] else None


def _norm(body):
    raw = json.dumps(body).encode()
    new_raw, new_body, stats = schemas.normalize_body(raw, body)
    return new_raw, new_body, stats


def _items_pattern(body):
    return body["tools"][0]["function"]["parameters"]["properties"]["a"]["items"]["pattern"]


# --- ancoragem (o pedido literal da tarefa) ---------------------------------
def test_unanchored_wrapped_both_sides():
    _, out, stats = _norm(_body("abc"))
    assert _items_pattern(out) == "^.*(?:abc).*$"
    assert ("anchor", "unanchored", 1) in stats.events


def test_start_anchor_only_completes_the_end():
    _, out, _ = _norm(_body("^https?:\\/\\/"))
    # ancora a ponta que faltava E normaliza o escape `\/` (o conversor nao traduz)
    assert _items_pattern(out) == "^https?://.*$"


def test_end_anchor_only_completes_the_start():
    _, out, _ = _norm(_body("abc$"))
    assert _items_pattern(out) == "^.*abc$"


def test_start_anchor_with_unsafe_char_class_after_escape_fix():
    _, out, _ = _norm(_body("^[a-z]+"))
    assert _items_pattern(out) == "^[a-z]+.*$"


def test_already_anchored_is_byte_identical():
    body = _body("^[a-z0-9][a-z0-9_-]*$")
    raw = json.dumps(body).encode()
    new_raw, new_body, stats = schemas.normalize_body(raw, body)
    assert (new_raw, new_body) == (raw, body)
    assert new_body is body and not stats.events


def test_body_without_patterns_is_byte_identical():
    body = {"model": "m", "messages": U,
            "tools": [{"type": "function", "function": {"name": "z", "parameters":
                      {"type": "object", "properties": {"q": {"type": "string"}}}}}],
            "response_format": {"type": "text"}}
    raw = "{\"model\":\"m\",\"messages\":[],\"tools\":[]}\n".encode()
    new_raw, new_body, stats = schemas.normalize_body(raw, body)
    assert new_raw is raw and new_body is body and not stats.events


def test_every_pattern_anchored_keeps_bytes_even_with_odd_spacing():
    # JSON com espacos/ordem "estranha": nada e' reescrito, entao os bytes voltam iguais
    raw = (b'{"messages": [{"role": "user", "content": "oi"}],'
           b' "tools": [{"type": "function", "function": {"name": "z", "parameters":'
           b' {"properties": {"a": {"items": {"pattern": "^x+$", "type": "string"},'
           b' "type": "array"}}}}}], "model": "pool"}')
    body = json.loads(raw)
    new_raw, _, stats = schemas.normalize_body(raw, body)
    assert new_raw is raw and not stats.events


def test_idempotent():
    _, once, _ = _norm(_body("abc"))
    _, twice, stats = _norm(once)
    assert _items_pattern(twice) == _items_pattern(once)
    assert not stats.events


# --- construtos que o conversor do llama.cpp tambem recusa ------------------
def test_escape_de_classe_e_descartado():
    _, out, stats = _norm(_body("^\\d+"))
    props = out["tools"][0]["function"]["parameters"]["properties"]["a"]["items"]
    assert "pattern" not in props
    assert ("drop", "escape-class", 1) in stats.events


def test_backreference_descartado():
    _, out, stats = _norm(_body("^(a)\\1$".replace("$", "")))
    assert "pattern" not in out["tools"][0]["function"]["parameters"]["properties"]["a"]["items"]
    assert any(a == "drop" and r == "backreference" for a, r, _n in stats.events)


def test_classe_posix_descartada():
    _, _, stats = _norm(_body("^[[:alpha:]]+"))
    assert any(a == "drop" and r == "posix-class" for a, r, _n in stats.events)


def test_flag_inline_descartada():
    _, _, stats = _norm(_body("(?i)^abc"))
    assert any(a == "drop" and r == "group-flag" for a, r, _n in stats.events)


def test_escape_seguro_sobrevive():
    _, out, _ = _norm(_body("a\\.b"))
    assert _items_pattern(out) == "^.*(?:a\\.b).*$"


def test_grupo_nao_capturante_sobrevive():
    _, out, _ = _norm(_body("^(?:a|b)"))
    assert _items_pattern(out) == "^(?:a|b).*$"


def test_anchored_but_unsafe_pattern_is_left_alone():
    # regra explicita: pattern JA ancorado nao e' tocado (bytes identicos), mesmo que o
    # conversor va recusar — o trafego que hoje funciona nao pode mudar
    body = _body("^\\d+$")
    raw = json.dumps(body).encode()
    assert schemas.normalize_body(raw, body)[:2] == (raw, body)


def test_repeticao_acima_do_teto_e_removida():
    body = _body("abc")
    params = body["tools"][0]["function"]["parameters"]["properties"]["a"]
    params["maxItems"] = 2000
    params["minItems"] = 4000
    _, out, stats = _norm(body)
    arr = out["tools"][0]["function"]["parameters"]["properties"]["a"]
    assert "maxItems" not in arr and "minItems" not in arr
    assert ("drop", "cap-maxItems", 1) in stats.events
    assert ("drop", "cap-minItems", 1) in stats.events


def test_maxlength_longo_e_removido():
    # medido: 1000 ok, 2000 -> 400 nos dois backends (o numero vira repeticao da gramatica)
    body = {"model": "pool", "messages": U, "tools": [{"type": "function", "function": {
        "name": "z", "parameters": {"type": "object", "required": ["o"], "properties": {
            "o": {"type": "object", "required": ["i"], "properties": {
                "i": {"type": "string", "maxLength": 10000}}}}}}}]}
    _, out, stats = _norm(body)
    assert "maxLength" not in out["tools"][0]["function"]["parameters"]["properties"]["o"]["properties"]["i"]
    assert ("drop", "cap-maxLength", 1) in stats.events


def test_repeticao_dentro_do_teto_fica():
    body = _body("^abc$")
    params = body["tools"][0]["function"]["parameters"]["properties"]["a"]
    params["maxItems"] = 1000
    raw = json.dumps(body).encode()
    assert schemas.normalize_body(raw, body)[:2] == (raw, body)


# --- outros lugares do corpo ------------------------------------------------
def test_response_format_json_schema():
    body = {"model": "pool", "messages": U, "response_format": {
        "type": "json_schema", "json_schema": {"name": "r", "schema": {
            "type": "object", "properties": {"x": {"type": "string", "pattern": "^https?:\\/\\/"}}}}}}
    _, out, stats = _norm(body)
    assert out["response_format"]["json_schema"]["schema"]["properties"]["x"]["pattern"] == \
        "^https?://.*$"
    assert ("anchor", "start_only", 1) in stats.events


def test_functions_legado():
    body = {"model": "pool", "messages": U, "functions": [
        {"name": "f", "parameters": {"type": "object", "properties": {
            "a": {"type": "array", "items": {"type": "string", "pattern": "abc"}}}}}]}
    _, out, _ = _norm(body)
    assert out["functions"][0]["parameters"]["properties"]["a"]["items"]["pattern"] == \
        "^.*(?:abc).*$"


def test_schemas_aninhados_anyof_e_defs():
    body = {"model": "pool", "messages": U, "tools": [{"type": "function", "function": {
        "name": "z", "parameters": {"type": "object", "$defs": {"S": {"type": "string",
                                                                     "pattern": "x+"}},
                                    "properties": {"a": {"anyOf": [
                                        {"type": "array", "items": {"type": "string",
                                                                    "pattern": "y+"}},
                                        {"$ref": "#/$defs/S"}]}}}}}], }
    _, out, _ = _norm(body)
    params = out["tools"][0]["function"]["parameters"]
    assert params["$defs"]["S"]["pattern"] == "^.*(?:x+).*$"
    assert params["properties"]["a"]["anyOf"][0]["items"]["pattern"] == "^.*(?:y+).*$"


def test_propriedade_chamada_pattern_e_dado_nao_sao_tocados():
    body = {"model": "pool", "messages": U, "tools": [{"type": "function", "function": {
        "name": "z", "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string", "pattern": "abc", "default": "zzz",
                        "examples": ["abc"]}}}}}], }
    _, out, _ = _norm(body)
    props = out["tools"][0]["function"]["parameters"]["properties"]["pattern"]
    # o schema da propriedade chamada "pattern" e' um schema: o pattern DELE e' ancorado...
    assert props["pattern"] == "^.*(?:abc).*$"
    # ...mas os DADOS (default/examples) ficam intactos
    assert props["default"] == "zzz" and props["examples"] == ["abc"]


def test_corpo_nao_dict():
    raw = b"[]"
    assert schemas.normalize_body(raw, []) == (raw, [], schemas.Stats())


# --- integracao: o que chega a engine --------------------------------------
async def test_engine_recebe_o_corpo_reparado(env):
    f = Fake("unico")
    p = await env([f], [{}])
    st, backend, _, _ = await p.chat(U, body={"tools": _body("abc")["tools"],
                                              "tool_choice": "auto"})
    assert (st, backend) == (200, "unico")
    got = f.requests[0]["body"]
    assert got["tools"][0]["function"]["parameters"]["properties"]["a"]["items"]["pattern"] == \
        "^.*(?:abc).*$"


async def test_engine_recebe_bytes_iguais_quando_ja_ancorado(env):
    f = Fake("unico")
    p = await env([f], [{}])
    body = {"model": "pool", "messages": U, "tool_choice": "auto",
            "tools": _body("^[a-z]+$")["tools"]}
    raw = json.dumps(body, indent=1, sort_keys=True).encode()
    async with p.http.post(p.url + "/v1/chat/completions", data=raw,
                           headers={"Content-Type": "application/json"}) as r:
        assert r.status == 200
    assert json.dumps(f.requests[0]["body"], indent=1,
                      sort_keys=True).encode() == raw


async def test_metrica_de_reparo(env):
    f = Fake("unico")
    p = await env([f], [{}])
    await p.chat(U, body={"tools": _body("abc")["tools"], "tool_choice": "auto"})
    txt = p.app.metrics.render(p.app.pool.states.values())
    assert 'pool_schema_normalizations_total{action="anchor",reason="unanchored"} 1' in txt
    assert 'pool_schema_normalizations_total{action="request",reason="rewritten"} 1' in txt


async def test_desligar_o_reparo_pela_config(env):
    f = Fake("unico")
    p = await env([f], [{}], schema_normalize=False)
    await p.chat(U, body={"tools": _body("abc")["tools"], "tool_choice": "auto"})
    assert f.requests[0]["body"]["tools"][0]["function"]["parameters"]["properties"]["a"]["items"][
        "pattern"] == "abc"
