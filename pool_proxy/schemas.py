"""Reparo dos JSON Schemas de ferramentas antes de encaminhar a engine.

O `json_schema_to_grammar` do llama.cpp (mainline na 3090 e o fork do MI50) recusa
`pattern` que nao comece com '^' e termine com '$':

    400 JSON schema conversion failed: Pattern must start with '^' and end with '$'

E, mesmo ancorado, o conversor estraga/invalida alguns construtos: escapes de classe
(`\\d`, `\\w`, `\\s` ...), `\\/`, `\\-`, `\\b`, `\\f`, `\\v`, backreference (`\\1`),
`\\p{..}`, flag inline (`(?i)`) e classe POSIX (`[[:alpha:]]` — esta chega a PENDURAR a
requisicao). Medido pedido a pedido contra dois backends vivos.

Regra deste modulo (a mesma para todo cliente, porque fica no caminho unico):

1. `pattern` ja ancorado (`^...$`) nao e tocado — nem o texto dele, nem o corpo.
2. `pattern` sem ancora vira `^.*(?:R).*$`; comecando com '^' vira `^R.*$`; terminando
   com '$' vira `^.*R`.
3. No pattern que NOS reescrevemos, `\\/` vira `/` (mesma semantica, GBNF valido).
4. Se depois disso o pattern ainda tem construto que o conversor recusa, o `pattern` e
   REMOVIDO (o resto do schema fica) — a gramatica fica mais frouxa, o pedido nao quebra.
5. `minItems`/`maxItems`/`minLength`/`maxLength` acima do teto medido (1000 cabe, 2000 nao)
   sao REMOVIDOS do schema (sem isso o conversor tambem devolve 400 — o numero vira
   repeticao na gramatica). Remover so afrouxa: nunca rejeita o que o cliente aceitaria.
6. Se nada precisa mudar, os BYTES originais sao devolvidos intactos (importante para o
   cache de prefixo e para o handoff, que casa o `apply-template` com o decode).

Nada de conteudo de prompt aqui: o modulo so olha `tools[].function.parameters`,
`functions[].parameters` e `response_format.json_schema.schema`, e a telemetria e' de
contagem (nunca o texto do pattern).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

# Chaves de um schema que levam a OUTROS schemas. Qualquer outra chave (`default`,
# `enum`, `const`, `description`, `examples`...) e dado: nao se desce nela.
_MAP_OF_SCHEMAS = ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas")
_SINGLE_SCHEMA = ("items", "additionalItems", "additionalProperties", "unevaluatedProperties",
                  "not", "contains", "propertyNames", "if", "then", "else")
_LIST_OF_SCHEMAS = ("anyOf", "oneOf", "allOf", "prefixItems")

# Escapes que o conversor do llama.cpp nao traduz para GBNF (viram gramatica invalida).
_ESC_CLASS = frozenset("dDsSwW")
_ESC_BOUNDARY = frozenset("bB")
_ESC_OTHER = frozenset("fvpPkKgGzZaA")
_DIGITS = frozenset("0123456789")
_GROUP_OK = ("(?:", "(?=", "(?!", "(?<=", "(?<!")

# Teto medido dos numeros que o conversor expande em REPETICAO de gramatica:
# `maxLength/minLength 1000` gera gramatica valida e `2000` devolve 400 nos dois backends;
# `maxItems 2000` e `minItems 4000` tambem devolvem 400. O keyword acima do teto e'
# REMOVIDO (gramatica mais frouxa) — clampar para baixo restringiria a saida do modelo.
REPEAT_KEYS = ("minItems", "maxItems", "minLength", "maxLength")
MINMAX_CAP = 1000


@dataclass
class Stats:
    """Contagens do reparo (nunca conteudo) para metrica/log."""
    events: list = field(default_factory=list)   # [(action, reason, n)]

    def add(self, action: str, reason: str, n: int = 1) -> None:
        for i, (a, r, c) in enumerate(self.events):
            if a == action and r == reason:
                self.events[i] = (a, r, c + n)
                return
        self.events.append((action, reason, n))

    @property
    def changed(self) -> bool:
        return bool(self.events) and any(a in ("anchor", "escape", "drop", "cap", "request")
                                        for a, _r, _n in self.events)

    def summary(self) -> str:
        return " ".join(f"{a}/{r}={n}" for a, r, n in self.events)


def _unsafe(pattern: str) -> str | None:
    """Motivo (rotulo curto) se o pattern provavelmente quebra o conversor, senao None."""
    i, n = 0, len(pattern)
    while i < n:
        c = pattern[i]
        if c == "\\":
            nxt = pattern[i + 1] if i + 1 < n else ""
            if not nxt:
                return "escape-final"
            if nxt in _ESC_CLASS:
                return "escape-class"
            if nxt in _ESC_BOUNDARY:
                return "escape-boundary"
            if nxt in _ESC_OTHER:
                return "escape-other"
            if nxt in _DIGITS:
                return "backreference"
            if nxt == "/":
                return "escape-slash"
            if nxt == "-":
                return "escape-hyphen"
            i += 2
            continue
        if c == "[" and pattern[i + 1:i + 3] == "[:":        # classe POSIX
            return "posix-class"
        if c == "(" and pattern[i + 1:i + 2] == "?":         # grupo com flags/condicional
            head = pattern[i:i + 4]
            if not any(head.startswith(ok) for ok in _GROUP_OK):
                return "group-flag"
        i += 1
    return None


def _fix_pattern(pattern: str) -> tuple[str | None, list]:
    """(pattern novo ou None se deve ser descartado, [(action, reason), ...]).

    `None` + lista vazia = pattern ja ancorado, nada a fazer.
    """
    anchored_start = pattern.startswith("^")
    anchored_end = pattern.endswith("$")
    if anchored_start and anchored_end:
        return pattern, []
    events: list = []
    if anchored_start:
        new = pattern + ".*$"
        events.append(("anchor", "start_only"))
    elif anchored_end:
        new = "^.*" + pattern
        events.append(("anchor", "end_only"))
    else:
        new = "^.*(?:" + pattern + ").*$"
        events.append(("anchor", "unanchored"))
    plain = new.replace("\\/", "/")
    if plain != new:
        events.append(("escape", "slash"))
        new = plain
    reason = _unsafe(new)
    if reason is not None:
        events.append(("drop", reason))
        return None, events
    return new, events


def _walk(node, stats: Stats, cap: int):
    """Percorre um schema devolvendo o MESMO objeto quando nada muda (copy-on-write)."""
    if isinstance(node, list):
        out = None
        for index, item in enumerate(node):
            new = _walk(item, stats, cap)
            if new is not item:
                if out is None:
                    out = list(node)
                out[index] = new
        return node if out is None else out
    if not isinstance(node, dict):
        return node

    out = None
    for key, value in node.items():
        if key == "pattern" and isinstance(value, str):
            fixed, events = _fix_pattern(value)
            if not events:
                continue
            if out is None:
                out = dict(node)
            for action, reason in events:
                stats.add(action, reason)
            if fixed is None:
                del out["pattern"]
            else:
                out["pattern"] = fixed
            continue
        if key in REPEAT_KEYS and type(value) is int and value > cap:
            if out is None:
                out = dict(node)
            del out[key]
            stats.add("drop", "cap-" + key)
            continue
        if key in _MAP_OF_SCHEMAS and isinstance(value, dict):
            new_map = None
            for name, sub in value.items():
                new_sub = _walk(sub, stats, cap)
                if new_sub is not sub:
                    if new_map is None:
                        new_map = dict(value)
                    new_map[name] = new_sub
            new = value if new_map is None else new_map
        elif key in _SINGLE_SCHEMA:
            new = _walk(value, stats, cap) if isinstance(value, (dict, list)) else value
        elif key in _LIST_OF_SCHEMAS and isinstance(value, list):
            new = _walk(value, stats, cap)
        else:
            continue
        if new is not value:
            if out is None:
                out = dict(node)
            out[key] = new
    return node if out is None else out


def _normalize_schema(schema, stats: Stats, cap: int):
    if not isinstance(schema, (dict, list)):
        return schema
    return _walk(schema, stats, cap)


def _tool_schemas(body: dict):
    """(campo, indice, schema) de cada schema de ferramenta do corpo."""
    for field_name in ("tools", "functions"):
        tools = body.get(field_name)
        if not isinstance(tools, list):
            continue
        for index, tool in enumerate(tools):
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
            if isinstance(fn, dict) and "parameters" in fn:
                yield field_name, index, fn, "parameters"


def normalize_body(raw: bytes, body: dict, *, cap: int = MINMAX_CAP) -> tuple[bytes, dict, Stats]:
    """Normaliza os schemas do corpo. Devolve `(raw, body, stats)` — os ORIGINAIS se nada muda."""
    stats = Stats()
    if not isinstance(body, dict):
        return raw, body, stats
    # caminho rapido: sem `pattern` e sem min/maxItems nao ha nada a reparar
    if (b'"pattern"' not in raw and b'"maxItems"' not in raw and b'"minItems"' not in raw
            and b'"maxLength"' not in raw and b'"minLength"' not in raw):
        return raw, body, stats

    out = body
    for _field, _index, fn, key in list(_tool_schemas(body)):
        new = _normalize_schema(fn.get(key), stats, cap)
        if new is not fn.get(key):
            if out is body:
                out = dict(body)
                out[_field] = list(body[_field])
                out[_field][_index] = dict(body[_field][_index])
            else:
                out[_field] = list(out[_field])
                out[_field][_index] = dict(out[_field][_index])
            target = out[_field][_index]
            if isinstance(target.get("function"), dict):
                target["function"] = dict(target["function"])
                target["function"][key] = new
            else:
                target[key] = new

    rf = body.get("response_format")
    if isinstance(rf, dict) and isinstance(rf.get("json_schema"), dict):
        js = rf["json_schema"]
        if "schema" in js:
            new = _normalize_schema(js.get("schema"), stats, cap)
            if new is not js.get("schema"):
                if out is body:
                    out = dict(body)
                out["response_format"] = dict(rf)
                out["response_format"]["json_schema"] = dict(js)
                out["response_format"]["json_schema"]["schema"] = new

    if out is body:
        return raw, body, stats
    stats.add("request", "rewritten")
    return json.dumps(out, ensure_ascii=False).encode(), out, stats
