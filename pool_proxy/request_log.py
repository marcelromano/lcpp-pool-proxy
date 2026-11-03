"""Best-effort, content-free request telemetry. Never blocks the event loop on I/O."""
import datetime
import json
import logging
import logging.handlers
import math
import os
import queue
import sys
import threading
import time

from aiohttp import web

from .paths import state_path

REQLOG_KEY = web.RequestKey('_reqlog', dict)
_log = logging.getLogger('pool_proxy.request_log')
# Aviso de descarte: no maximo um log a cada intervalo; o contador de descartes segue exato.
DROP_LOG_INTERVAL_S = 60.0
_drop_lock = threading.Lock()
_drops = {'queue_full': 0, 'write_error': 0}
_drops_unlogged = 0
_last_drop_log = None

_FIELDS = ('top_k', 'top_p', 'temperature', 'min_p', 'typical_p', 'tfs_z',
           'repeat_penalty', 'repeat_last_n', 'presence_penalty', 'frequency_penalty',
           'mirostat', 'mirostat_tau', 'mirostat_eta', 'dynatemp_range',
           'dynatemp_exponent', 'xtc_probability', 'xtc_threshold', 'n_probs',
           'seed', 'n', 'max_tokens', 'max_completion_tokens', 'n_predict')
_PRESENT = ('grammar', 'json_schema', 'logit_bias', 'response_format', 'samplers')
# knobs de raciocinio. Sao registrados como valor curto/bool, NUNCA com prompt/mensagem:
# a pergunta operacional e' "o que o cliente mandou chegou ao pool?" (propagacao, nao politica).
_THINK_TOP = ('reasoning_effort', 'enable_thinking', 'reasoning_budget', 'thinking_budget',
              'thinking_token_budget', 'preserve_thinking')
_THINK_CTK = ('enable_thinking', 'reasoning_effort', 'reasoning_budget', 'thinking_budget',
              'thinking_token_budget', 'preserve_thinking')
_TIMINGS = ('prompt_n', 'prompt_ms', 'prompt_per_token_ms', 'prompt_per_second',
            'predicted_n', 'predicted_ms', 'predicted_per_token_ms',
            'predicted_per_second', 'cache_n')
_queue = queue.Queue(maxsize=2048)


def _knob(v):
    """Valor de knob de raciocinio, so se for escalar curto (bool/numero/string curta)."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and _number(v):
        return v
    if isinstance(v, str) and len(v) <= 24:
        return v
    return None


def _thinking(data):
    think = {}
    for k in _THINK_TOP:
        if k in data:
            v = _knob(data[k])
            if v is not None:
                think[k] = v
    ctk = data.get('chat_template_kwargs')
    if isinstance(ctk, dict):
        for k in _THINK_CTK:
            if k in ctk:
                v = _knob(ctk[k])
                if v is not None:
                    think['ctk.' + k] = v
    th = data.get('thinking')
    if isinstance(th, dict):
        t, b = _knob(th.get('type')), _knob(th.get('budget_tokens'))
        if t is not None:
            think['thinking.type'] = t
        if b is not None:
            think['thinking.budget_tokens'] = b
    return think


def _number(v):
    return type(v) in (int, float) and (type(v) is int or math.isfinite(v)) and abs(v) < 1e100


def request_log_path():
    """POOL_REQUEST_LOG, senao <estado>/request-log.jsonl (POOL_PROXY_STATE_DIR ou /var/lib/pool-proxy)."""
    return os.environ.get('POOL_REQUEST_LOG', '').strip() or state_path('request-log.jsonl')


def drops():
    """Contadores de descarte desde o inicio do processo: fila cheia e falha de escrita."""
    with _drop_lock:
        return dict(_drops)


def _record_drop(reason, exc=None):
    global _drops_unlogged, _last_drop_log
    now = time.monotonic()
    with _drop_lock:
        _drops[reason] += 1
        _drops_unlogged += 1
        if _last_drop_log is not None and now - _last_drop_log < DROP_LOG_INTERVAL_S:
            return
        unlogged, _drops_unlogged = _drops_unlogged, 0
        _last_drop_log = now
        total = dict(_drops)
    _log.error('request-log: %d registro(s) descartado(s) desde o ultimo aviso (total: fila_cheia=%d, '
               'falha_escrita=%d); arquivo=%s; ultimo erro=%s: %s. Defina POOL_REQUEST_LOG ou '
               'POOL_PROXY_STATE_DIR para um diretorio gravavel.',
               unlogged, total['queue_full'], total['write_error'], request_log_path(),
               type(exc).__name__ if exc is not None else reason, exc if exc is not None else '')


class _CountingHandler(logging.handlers.RotatingFileHandler):
    """logging engole erro de escrita em handleError; aqui ele vira descarte contado e logado."""

    def handleError(self, record):
        _record_drop('write_error', sys.exc_info()[1])


def _emit(state, record):
    """Escreve um registro. Erro de abertura propaga para o chamador, que conta o descarte."""
    try:
        if state['handler'] is None:
            state['handler'] = _CountingHandler(request_log_path(), maxBytes=10*1024*1024,
                                                backupCount=5, encoding='utf-8', delay=True)
        state['handler'].emit(logging.makeLogRecord({'msg': json.dumps(record, separators=(',', ':'), allow_nan=False),
                                                     'levelno': logging.INFO, 'levelname': 'INFO'}))
    except Exception as exc:
        _close(state)
        _record_drop('write_error', exc)


def _close(state):
    handler = state['handler']
    state['handler'] = None
    if handler is not None:
        try:
            handler.close()
        except Exception:
            pass


def _run():
    state = {'handler': None}
    while True:
        record = _queue.get()
        try:
            _emit(state, record)
        finally:
            _queue.task_done()


def start():
    try:
        threading.Thread(target=_run, name='request-log', daemon=True).start()
    except Exception:
        pass


def begin(request):
    try:
        request[REQLOG_KEY] = {'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                              '_start': time.monotonic(), 'model': None, 'backend': None,
                              'prompt_tokens': None, 'generation_tokens': None,
                              'slot': None, 'timings': {}, 'sampling': {}}
    except Exception:
        pass


def body(request, data, models):
    try:
        rec = request[REQLOG_KEY]
        model = data.get('model')
        if isinstance(model, str) and model in models:
            rec['model'] = model
        rec['sampling'] = {k: data[k] for k in _FIELDS if k in data and _number(data[k])}
        rec['sampling'].update({k + '_present': bool(data[k]) for k in _PRESENT if k in data})
        if isinstance(data.get('stream'), bool):
            rec['stream'] = data['stream']
        think = _thinking(data)
        if think:
            rec['thinking'] = think
    except Exception:
        pass


def backend(request, name):
    try:
        request[REQLOG_KEY]['backend'] = name
    except Exception:
        pass


def schema(request, events):
    """contagens do reparo de JSON Schema antes do forward. Nunca o texto do pattern."""
    try:
        ev = {f'{action}:{reason}': n for action, reason, n in events}
        if ev:
            request[REQLOG_KEY]['schema'] = ev
    except Exception:
        pass


def response(request, raw):
    try:
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            return
        rec = request[REQLOG_KEY]
        usage = obj.get('usage')
        if isinstance(usage, dict):
            for source, dest in (('prompt_tokens', 'prompt_tokens'), ('completion_tokens', 'generation_tokens')):
                v = usage.get(source)
                if _number(v) and v >= 0:
                    rec[dest] = v
        timings = obj.get('timings')
        if isinstance(timings, dict):
            for k in _TIMINGS:
                if _number(timings.get(k)):
                    rec['timings'][k] = timings[k]
            # prompt_n is only newly evaluated tokens; cache_n is the reused prefix.
            if (rec['prompt_tokens'] is None and _number(timings.get('prompt_n'))
                    and _number(timings.get('cache_n'))):
                rec['prompt_tokens'] = timings['prompt_n'] + timings['cache_n']
            if rec['generation_tokens'] is None and _number(timings.get('predicted_n')):
                rec['generation_tokens'] = timings['predicted_n']
        for k in ('id_slot', 'slot_id'):
            if _number(obj.get(k)):
                rec['slot'] = obj[k]
    except Exception:
        pass


class SSE:
    def __init__(self, request):
        self.request = request
        self.buffer = b''
        self.discard = False

    def feed(self, chunk):
        try:
            # Only keep a bounded partial line. Oversized lines are discarded until newline.
            for part in chunk.splitlines(keepends=True):
                end = part.endswith(b'\n')
                if not self.discard:
                    if len(self.buffer) + len(part) > 65536:
                        self.buffer = b''
                        self.discard = True
                    else:
                        self.buffer += part
                if end:
                    if not self.discard and self.buffer.startswith(b'data:'):
                        response(self.request, self.buffer[5:].strip())
                    self.buffer = b''
                    self.discard = False
        except Exception:
            self.buffer = b''
            self.discard = True


def finish(request, status):
    try:
        rec = request[REQLOG_KEY]
        rec['duration_ms'] = round((time.monotonic() - rec.pop('_start')) * 1000, 2)
        rec['status'] = int(status)
    except Exception:
        return
    try:
        _queue.put_nowait(rec)
    except queue.Full:
        _record_drop('queue_full')
