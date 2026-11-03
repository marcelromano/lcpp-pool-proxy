import asyncio

from aiohttp import web

from conftest import Fake
from pool_proxy.probe import usable_prefix

BLK = 256


def fnv_blocks(ids, block=BLK):
    h, out = 0xcbf29ce484222325, []
    for i, t in enumerate(ids):
        v = t & 0xffffffff
        for s in (0, 8, 16, 24):
            h = ((h ^ ((v >> s) & 0xff)) * 0x100000001b3) & 0xffffffffffffffff
        if (i + 1) % block == 0:
            out.append("%016x" % h)
    return out


def render(body):
    return "".join(f"<{m['role']}>{m['content']}" for m in body["messages"]) + "<assistant>"


def tok(text):          # 1 caractere = 1 token
    return [ord(c) for c in text]


class Exp24Fake(Fake):
    """llama-server falso com patch slot-telemetry: /slots, /apply-template, /tokenize, input_tokens."""

    def __init__(self, name, nslots=2, input_tokens=True, slots_delay=0.0, **kw):
        super().__init__(name, **kw)
        self.slots = [{"ids": [], "ckpt": -1, "nck": 0} for _ in range(nslots)]
        self.input_tokens, self.slots_delay = input_tokens, slots_delay
        self.calls: dict[str, int] = {}

    def extra_routes(self, app):
        app.router.add_get("/slots", self._slots)
        app.router.add_post("/apply-template", self._apply)
        app.router.add_post("/tokenize", self._tokenize)
        app.router.add_post("/v1/chat/completions/input_tokens", self._input_tokens)

    def _n(self, k):
        self.calls[k] = self.calls.get(k, 0) + 1

    async def _slots(self, request):
        self._n("slots")
        await asyncio.sleep(self.slots_delay)
        out = []
        for i, s in enumerate(self.slots):
            out.append({"id": i, "is_processing": False, "n_cached_tokens": len(s["ids"]),
                        "n_checkpoints": s["nck"], "ckpt_pos_min": -1, "ckpt_pos_max": s["ckpt"],
                        "has_media": False, "prefix_hash_block": BLK,
                        "prefix_hash": fnv_blocks(s["ids"])})
        return web.json_response(out)

    async def _apply(self, request):
        self._n("apply")
        return web.json_response({"prompt": render(await request.json())})

    async def _tokenize(self, request):
        self._n("tokenize")
        b = await request.json()
        ids = tok(b["content"])
        return web.json_response({"tokens": ids, "ids": ids, "prefix_hash_block": BLK,
                                  "prefix_hash": fnv_blocks(ids)})

    async def _input_tokens(self, request):
        self._n("input_tokens")
        if not self.input_tokens:
            return web.json_response({"error": "nope"}, status=404)
        ids = tok(render(await request.json()))
        return web.json_response({"input_tokens": len(ids), "ids": ids, "has_media": False,
                                  "prefix_hash_block": BLK, "prefix_hash": fnv_blocks(ids)})

    async def _chat(self, request):
        body = await request.json()
        ids = tok(render(body))
        sl = self.slots[body.get("id_slot", 0)]
        n = 0
        for x, y in zip(sl["ids"], ids):
            if x != y:
                break
            n += 1
        sl["ids"] = ids + tok("ok")
        self.requests.append({"headers": dict(request.headers), "body": body})
        return web.json_response({"backend": self.name, "choices": [], "timings": {"cache_n": n}})


def conv(n_chars, tail="q"):
    return [{"role": "system", "content": "s" * n_chars}, {"role": "user", "content": tail}]


def prime(fake, slot, body_msgs):
    fake.slots[slot]["ids"] = tok(render({"messages": body_msgs})) + tok("ok")


def follow(msgs, q):
    return msgs + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": q}]


async def test_routes_to_slot_owner_and_sends_id_slot(env):
    a, b = Exp24Fake("a"), Exp24Fake("b")
    # b e pior em decode, mas tem o prefixo no slot 1
    p = await env([a, b], [{"decode_tps": 100, "prefill_tps": 50}, {"decode_tps": 30, "prefill_tps": 50}], probe_slots_ttl_s=0)
    base = conv(2000, "q1")
    prime(b, 1, base)
    st, backend, reason, _ = await p.chat(follow(base, "q2"), body={"max_tokens": 16})
    assert (st, backend, reason) == (200, "b", "affinity")
    assert b.requests[-1]["body"]["id_slot"] == 1
    assert p.app.probe.caps == {"a": True, "b": True}
    m = await (await p.http.get(p.url + "/metrics")).text()
    assert 'pool_prefix_reused_tokens_estimated_total{backend="b"}' in m
    assert 'pool_affinity_route_total{outcome="owner"} 1' in m
    assert 'pool_affinity_result_total{backend="b",result="hit"} 1' in m   # cache_n observado
    assert 'pool_probe_seconds_count{stage="slots"}' in m and 'stage="tokenize"' in m


async def test_cold_prompt_uses_cost_and_no_id_slot(env):
    a, b = Exp24Fake("a"), Exp24Fake("b")
    p = await env([a, b], [{"decode_tps": 30}, {"decode_tps": 100}], probe_slots_ttl_s=0)
    st, backend, reason, _ = await p.chat(conv(1500))
    assert (st, backend, reason) == (200, "b", "cost")
    assert "id_slot" not in b.requests[-1]["body"]


async def test_apply_template_fallback_and_lru_cache(env):
    a = Exp24Fake("a", input_tokens=False)
    p = await env([a], [{}], probe_slots_ttl_s=0)
    msgs = conv(600)
    prime(a, 0, msgs)
    await p.chat(follow(msgs, "x"))
    assert a.calls.get("apply") == 1 and a.calls.get("tokenize") == 1
    assert a.requests[-1]["body"]["id_slot"] == 0
    # mesmo corpo de mensagens: tokenizacao vem do LRU (so /slots e consultado de novo)
    await p.chat(follow(msgs, "x"))
    assert a.calls.get("apply") == 1 and a.calls.get("tokenize") == 1
    assert a.calls["input_tokens"] == 1          # modo lembrado: nao reinsiste no endpoint ausente


async def test_slots_timeout_is_ignored_not_fatal(env):
    a, b = Exp24Fake("a", slots_delay=2.0), Exp24Fake("b")
    p = await env([a, b], [{"decode_tps": 100, "prefill_tps": 50}, {"decode_tps": 30, "prefill_tps": 50}], probe_slots_timeout_s=0.2, probe_slots_ttl_s=0)
    base = conv(1000)
    prime(b, 0, base)
    st, backend, reason, _ = await p.chat(follow(base, "z"), body={"max_tokens": 8})
    assert st == 200 and backend == "b"       # a ignorado no probe; b tem afinidade
    assert p.app.probe.caps.get("a") is None
    m = await (await p.http.get(p.url + "/metrics")).text()
    assert 'pool_probe_errors_total{stage="slots",backend="a"}' in m


async def test_backend_without_slot_telemetry_falls_back_to_mirror(env):
    plain, x = Fake("plain"), Exp24Fake("x")
    p = await env([plain, x], [{"decode_tps": 100}, {"decode_tps": 30}], probe_slots_ttl_s=0)
    first = conv(1500, "q1")
    _, b1, _, _ = await p.chat(first)
    assert b1 == "plain" and p.app.probe.caps["plain"] is False and p.app.probe.caps["x"] is True
    p.app.pool.states["plain"].avg_s = 3.0       # fila fabricada: custo do plain sobe, espelho compensa
    nxt = first + [{"role": "assistant", "content": "r"}, {"role": "user", "content": "q2"}]
    _, b2, reason, _ = await p.chat(nxt)
    assert (b2, reason) == ("plain", "affinity")


async def test_all_plain_backends_keep_old_behaviour(env):
    a, b = Fake("a"), Fake("b")
    p = await env([a, b], [{"decode_tps": 30}, {"decode_tps": 68}], probe_slots_ttl_s=0)
    st, backend, reason, _ = await p.chat([{"role": "user", "content": "oi"}])
    assert (st, backend, reason) == (200, "b", "cost")
    assert not p.app.probe.caps["a"] and not p.app.probe.caps["b"]


async def test_client_id_slot_is_not_overridden(env):
    a = Exp24Fake("a")
    p = await env([a], [{}], probe_slots_ttl_s=0)
    base = conv(800)
    prime(a, 1, base)
    await p.chat(follow(base, "n"), body={"id_slot": 0})
    assert a.requests[-1]["body"]["id_slot"] == 0


def test_usable_prefix_rules():
    ids = list(range(1000))
    sh = fnv_blocks(ids)                      # 3 blocos
    slot = {"prefix_hash": sh, "n_cached_tokens": 1000, "n_checkpoints": 0, "ckpt_pos_max": -1}
    ext = fnv_blocks(ids + list(range(5000, 5300)))
    # pedido estende todo o cache: usa n_cached
    assert usable_prefix(slot, ext, 1300) == 1000
    # diverge no 3o bloco, sem checkpoints: blocos iguais x 256
    div = fnv_blocks(ids[:600] + [9] * 700)
    assert usable_prefix(slot, div, 1300) == 512
    # hibrido: so ha estado ate ckpt_pos_max+1
    slot_h = {**slot, "n_checkpoints": 2, "ckpt_pos_max": 299}
    assert usable_prefix(slot_h, div, 1300) == 300
    # hibrido sem checkpoint valido
    assert usable_prefix({**slot, "n_checkpoints": 1, "ckpt_pos_max": -1}, div, 1300) == 0
    # hibrido mas o pedido estende o cache inteiro: estado vivo vale
    assert usable_prefix(slot_h, ext, 1300) == 1000
    # prompt identico: reprocessa ao menos 1 token
    assert usable_prefix(slot, sh, 768) == 767
    # slot sem bloco completo
    assert usable_prefix({"prefix_hash": [], "n_cached_tokens": 100}, ext, 1300) == 0
