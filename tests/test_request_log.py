import json
import logging
import os
import queue
import types

import pytest
from aiohttp import web

from pool_proxy import paths, request_log as rl


def test_sanitized_response_and_stream_fragments():
    request = {}
    rl.begin(request)
    rl.body(request, {'model': 'local/pool', 'messages': [{'content': 'SECRET_PROMPT'}],
                      'temperature': 0.25, 'top_k': 7, 'grammar': 'SECRET_GRAMMAR',
                      'logit_bias': {'SECRET_BIAS': 1}}, ['local/pool'])
    rl.backend(request, 'rtx3090-27b')
    s = rl.SSE(request)
    for chunk in (b'data: {"choices":[{"content":"SECRET_RESPONSE"}]}\n',
                  b'data: {"tim', b'ings":{"prompt_n":4,"cache_n":58,',
                  b'"predicted_n":8,"predicted_ms":120}}\n\n'):
        s.feed(chunk)
    rec = request[rl.REQLOG_KEY]
    assert rec['prompt_tokens'] == 62
    assert rec['generation_tokens'] == 8
    assert rec['sampling']['top_k'] == 7
    assert rec['sampling']['grammar_present'] is True
    assert 'SECRET' not in json.dumps(rec)


def test_usage_overrides_timings_and_invalid_model_is_not_logged():
    request = {}
    rl.begin(request)
    rl.body(request, {'model': 'SECRET_MODEL', 'top_p': 'SECRET', 'min_p': 0.1}, ['local/pool'])
    rl.response(request, b'{"usage":{"prompt_tokens":42,"completion_tokens":3},"timings":{"prompt_n":1,"cache_n":2}}')
    rec = request[rl.REQLOG_KEY]
    assert rec['model'] is None
    assert rec['prompt_tokens'] == 42 and rec['generation_tokens'] == 3
    assert rec['sampling'] == {'min_p': 0.1}
    assert 'SECRET' not in json.dumps(rec)


def test_bounded_sse_discards_oversized_line():
    request = {}
    rl.begin(request)
    s = rl.SSE(request)
    s.feed(b'data: ' + b'x' * 70000)
    s.feed(b'continued\ndata: {"usage":{"completion_tokens":5}}\n')
    assert request[rl.REQLOG_KEY]['generation_tokens'] == 5
    assert len(s.buffer) == 0


def test_thinking_knobs_are_logged_without_content():
    """prova de propagacao. Os knobs de raciocinio chegam ao pool e aparecem no log;
    prompt/mensagem/conteudo NUNCA aparecem."""
    request = {}
    rl.begin(request)
    rl.body(request, {'model': 'local/pool',
                      'messages': [{'role': 'user', 'content': 'SEGREDO_DO_USUARIO'}],
                      'reasoning_effort': 'high', 'reasoning_budget': 64,
                      'chat_template_kwargs': {'enable_thinking': False, 'reasoning_effort': 'low',
                                               'preserve_thinking': True, 'texto_livre': 'SEGREDO'},
                      'thinking': {'type': 'disabled', 'budget_tokens': 16}}, ['local/pool'])
    rec = request[rl.REQLOG_KEY]
    assert rec['thinking'] == {
        'reasoning_effort': 'high', 'reasoning_budget': 64,
        'ctk.enable_thinking': False, 'ctk.reasoning_effort': 'low', 'ctk.preserve_thinking': True,
        'thinking.type': 'disabled', 'thinking.budget_tokens': 16}
    assert 'SEG' not in json.dumps(rec)


def test_sem_knobs_nao_cria_campo_thinking():
    request = {}
    rl.begin(request)
    rl.body(request, {'model': 'local/pool', 'messages': [{'role': 'user', 'content': 'oi'}]},
            ['local/pool'])
    assert 'thinking' not in request[rl.REQLOG_KEY]


@pytest.fixture
def clean_drops(monkeypatch):
    monkeypatch.setattr(rl, '_drops', {'queue_full': 0, 'write_error': 0})
    monkeypatch.setattr(rl, '_drops_unlogged', 0)
    monkeypatch.setattr(rl, '_last_drop_log', None)
    return rl._drops


def test_request_key_is_typed_and_not_a_str_key():
    assert isinstance(rl.REQLOG_KEY, web.RequestKey)
    request = {}
    rl.begin(request)
    assert rl.REQLOG_KEY in request
    assert '_reqlog' not in request


def test_request_log_path_resolution(monkeypatch, tmp_path):
    monkeypatch.delenv('POOL_PROXY_STATE_DIR', raising=False)
    monkeypatch.delenv('POOL_REQUEST_LOG', raising=False)
    assert rl.request_log_path() == os.path.join(paths.DEFAULT_STATE_DIR, 'request-log.jsonl')
    monkeypatch.setenv('POOL_PROXY_STATE_DIR', str(tmp_path))
    assert rl.request_log_path() == os.path.join(str(tmp_path), 'request-log.jsonl')
    monkeypatch.setenv('POOL_PROXY_STATE_DIR', '   ')
    assert paths.state_dir() == paths.DEFAULT_STATE_DIR
    monkeypatch.setenv('POOL_REQUEST_LOG', str(tmp_path / 'x.jsonl'))
    assert rl.request_log_path() == str(tmp_path / 'x.jsonl')
    monkeypatch.setenv('POOL_REQUEST_LOG', '')
    assert rl.request_log_path() == os.path.join(paths.DEFAULT_STATE_DIR, 'request-log.jsonl')


def test_write_failure_is_logged_and_counted_not_silent(monkeypatch, tmp_path, caplog, clean_drops):
    blocked = tmp_path / 'diretorio'
    blocked.mkdir()  # abrir um diretorio como arquivo falha na escrita
    monkeypatch.setenv('POOL_REQUEST_LOG', str(blocked))
    state = {'handler': None}
    with caplog.at_level(logging.ERROR, logger='pool_proxy.request_log'):
        rl._emit(state, {'a': 1})
        rl._emit(state, {'a': 2})
    assert clean_drops['write_error'] == 2
    errors = [r for r in caplog.records if r.name == 'pool_proxy.request_log']
    assert len(errors) == 1  # rate-limit: o segundo descarte entra no contador, nao num novo log
    assert 'POOL_REQUEST_LOG' in errors[0].getMessage()


def test_handler_error_path_is_counted(monkeypatch, tmp_path, caplog, clean_drops):
    monkeypatch.setenv('POOL_REQUEST_LOG', str(tmp_path / 'ok.jsonl'))
    handler = rl._CountingHandler(str(tmp_path / 'ok.jsonl'), encoding='utf-8', delay=True)
    with caplog.at_level(logging.ERROR, logger='pool_proxy.request_log'):
        try:
            raise OSError('disco cheio')
        except OSError:
            handler.handleError(logging.makeLogRecord({'msg': 'x'}))
    handler.close()
    assert clean_drops['write_error'] == 1
    assert any('disco cheio' in r.getMessage() for r in caplog.records)


def test_drop_log_is_rate_limited_and_counter_stays_exact(monkeypatch, caplog, clean_drops):
    now = [1000.0]
    monkeypatch.setattr(rl, 'time', types.SimpleNamespace(monotonic=lambda: now[0]))
    with caplog.at_level(logging.ERROR, logger='pool_proxy.request_log'):
        for _ in range(5):
            rl._record_drop('queue_full')          # 1 log imediato, 4 acumulados
        now[0] = 1030.0
        rl._record_drop('queue_full')              # dentro do intervalo: sem log
        now[0] = 1061.0
        rl._record_drop('write_error', OSError('falha'))  # passou 60 s: 1 log com 6 acumulados
    msgs = [r.getMessage() for r in caplog.records if r.name == 'pool_proxy.request_log']
    assert len(msgs) == 2
    assert '1 registro(s) descartado(s)' in msgs[0]
    assert '6 registro(s) descartado(s)' in msgs[1]
    assert rl.drops() == {'queue_full': 6, 'write_error': 1}


def test_full_queue_counts_drop(monkeypatch, clean_drops):
    q = queue.Queue(maxsize=1)
    q.put_nowait('ocupado')
    monkeypatch.setattr(rl, '_queue', q)
    request = {}
    rl.begin(request)
    rl.finish(request, 200)
    assert clean_drops['queue_full'] == 1
    assert q.qsize() == 1
    assert rl.drops()['queue_full'] == 1
