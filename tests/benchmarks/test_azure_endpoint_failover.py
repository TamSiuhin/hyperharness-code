import json
import threading
from http.server import ThreadingHTTPServer

import httpx
import pytest

from scripts.azure_endpoint_failover import EndpointPool, Handler, HOSTS, MODEL
from scripts.azure_endpoint_failover import completed_body, error_metadata


@pytest.mark.parametrize('failure', ['timeout', 'rate_limit', 'server', 'deployment', 'bad_request'])
def test_failover_preserves_request_and_never_logs_credentials(tmp_path, failure):
    calls = []

    def upstream(request):
        calls.append(request)
        if len(calls) == 1:
            if failure == 'timeout':
                raise httpx.ReadTimeout('private error contents', request=request)
            status = {'rate_limit': 429, 'server': 503, 'deployment': 404, 'bad_request': 400}[failure]
            return httpx.Response(status, json={'error': {'code': 'DeploymentNotFound'}})
        return httpx.Response(200, json={'choices': [{'message': {'content': 'OK'}}]})

    token = 'loopback-test-' + 'x' * 40
    pool = EndpointPool([{'url': 'https://' + host, 'key': 'private-' + str(index)}
        for index, host in enumerate(HOSTS)], token, tmp_path / 'events.jsonl',
        client=httpx.Client(transport=httpx.MockTransport(upstream)))
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.pool = pool
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        response = httpx.post(f'http://127.0.0.1:{server.server_port}/openai/responses?api-version=2025-04-01-preview',
            headers={'api-key': token}, json={'model': MODEL, 'input': 'hello'}, timeout=5)
        assert response.status_code == (400 if failure == 'bad_request' else 200)
        assert len(calls) == (1 if failure == 'bad_request' else 2)
        if len(calls) == 2:
            assert calls[0].content == calls[1].content
            assert calls[0].url.host == HOSTS[0] and calls[1].url.host == HOSTS[1]
            assert calls[1].headers['api-key'] == 'private-1'
        server.shutdown()
        thread.join()
        logs = (tmp_path / 'events.jsonl').read_text()
        assert 'private-' not in logs and token not in logs and 'hello' not in logs
        assert len([json.loads(line) for line in logs.splitlines()]) == len(calls)
    finally:
        server.shutdown()
        server.server_close()
        pool.client.close()


def test_pool_skips_failed_endpoints_and_rejects_unregistered_hosts(tmp_path):
    endpoints = [{'url': 'https://' + host, 'key': 'secret'} for host in HOSTS]
    pool = EndpointPool(endpoints, 'x' * 40, tmp_path / 'events.jsonl')
    try:
        assert pool.order() == [0, 1, 2]
        pool.failed(0)
        assert pool.order() == [1, 2]
        pool.failed(1)
        assert pool.order() == [2]
        pool.failed(2)
        assert pool.order() == []
        with pytest.raises(ValueError):
            EndpointPool([{'url': 'http://localhost', 'key': 'secret'}] * 3, 'x' * 40, tmp_path / 'bad')
    finally:
        pool.client.close()


@pytest.mark.parametrize('partial', [True, False])
def test_stream_failure_never_stitches_and_all_endpoints_are_bounded(tmp_path, partial):
    calls = []

    class BrokenStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b'data: {"type":"response.created"}\n\n'
            raise httpx.ReadTimeout('stream stalled')

    def upstream(request):
        calls.append(request.url.host)
        if partial:
            if len(calls) > 1:
                return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                    content=b'data: {"type":"response.completed"}\n\n')
            return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=BrokenStream())
        return httpx.Response(503, json={'error': {'message': 'unavailable'}})

    token = 'x' * 40
    pool = EndpointPool([{'url': 'https://' + host, 'key': 'private'} for host in HOSTS],
        token, tmp_path / 'events.jsonl', client=httpx.Client(transport=httpx.MockTransport(upstream)),
        max_attempts=3, request_budget=0.1)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.pool = pool
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f'http://127.0.0.1:{server.server_port}/openai/responses'
        response = httpx.post(url, headers={'api-key': token}, json={'model': MODEL, 'stream': True}, timeout=5)
        assert calls == list(HOSTS[:2] if partial else HOSTS)
        assert response.status_code == (200 if partial else 503)
        if partial:
            assert response.content == b'data: {"type":"response.completed"}\n\n'
        else:
            assert httpx.post(url, headers={'api-key': token}, json={'model': MODEL}, timeout=5).status_code == 503
            assert len(calls) == 3
        assert httpx.post(url, json={'model': MODEL}, timeout=5).status_code == 401
        assert httpx.post(url, headers={'api-key': token}, json={'model': 'wrong'}, timeout=5).status_code == 400
        assert httpx.post(url + '/wrong', headers={'api-key': token}, json={'model': MODEL}, timeout=5).status_code == 400
        server.shutdown()
        thread.join()
        records = [json.loads(line) for line in (tmp_path / 'events.jsonl').read_text().splitlines()]
        assert records[-1]['outcome'] == ('completed' if partial else 'retryable_status')
    finally:
        server.shutdown()
        server.server_close()
        pool.client.close()


def test_retry_waits_for_cooldown_and_has_attempt_and_time_caps(tmp_path):
    current = [100.0]
    waits = []

    def wait(seconds):
        waits.append(seconds)
        current[0] += seconds

    pool = EndpointPool([{'url': 'https://' + host, 'key': 'secret'} for host in HOSTS],
        'x' * 40, tmp_path / 'events.jsonl', clock=lambda: current[0], wait=wait,
        cooldown=2, max_attempts=6, request_budget=10)
    try:
        visited = []
        for attempt, index, remaining in pool.attempts('call'):
            visited.append(index)
            pool.failed(index, retry_after=3)
        assert visited == [0, 1, 2, 0, 1, 2]
        assert waits == [3]
        pool.failed(0, retry_after=50)
        pool.failed(1, retry_after=50)
        pool.failed(2, retry_after=50)
        assert list(pool.attempts('exhausted')) == []
        assert waits == [3]
    finally:
        pool.client.close()


def test_buffer_rejects_clean_eof_without_completion_and_redacts_error():
    import time
    response = httpx.Response(200, content=b'data: {"type":"response.created"}\n\n')
    with pytest.raises(httpx.RemoteProtocolError, match='response.completed'):
        completed_body(response, True, time.monotonic() + 10)
    response = httpx.Response(400, json={'error': {'code': 'invalid_encrypted_content',
        'message': 'sensitive prompt contents', 'param': 'input[0].encrypted_content', 'type': 'invalid_request_error'}})
    details = error_metadata(response)
    assert details['code'] == 'invalid_encrypted_content'
    assert 'sensitive' not in json.dumps(details) and 'message' not in details


def test_single_deployed_endpoint_retries_only_that_endpoint(tmp_path):
    current = [100.0]
    def wait(seconds):
        current[0] += seconds
    pool = EndpointPool([{'url': 'https://' + host, 'key': 'secret'} for host in HOSTS], 'x' * 40,
        tmp_path / 'events.jsonl', enabled_endpoints=[HOSTS[0]],
        clock=lambda: current[0], wait=wait, cooldown=2, request_budget=30)
    try:
        seen = []
        for attempt, index, remaining in pool.attempts('call'):
            seen.append(index)
            pool.failed(index)
        assert seen == [0] * 6
        assert current[0] == 110
        pool.missing_deployment(0)
        assert list(pool.attempts('no-deployment')) == []
    finally:
        pool.client.close()


def test_missing_deployments_never_return_after_cooldown(tmp_path):
    current = [100.0]
    pool = EndpointPool([{'url': 'https://' + host, 'key': 'secret'} for host in HOSTS],
        'x' * 40, tmp_path / 'events.jsonl', clock=lambda: current[0])
    try:
        pool.missing_deployment(1)
        pool.missing_deployment(2)
        pool.failed(0)
        assert pool.order() == []
        current[0] += 1000
        assert pool.order() == [0]
    finally:
        pool.client.close()