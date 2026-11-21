"""Loopback-only Azure failover for chat completions and streaming Responses."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import httpx


HOSTS = (
    'roar-dev-swedencentral.openai.azure.com',
    'roar-dev-northcentralus.openai.azure.com',
    'oar-oai-eastus2.openai.azure.com',
)
MODEL = 'gpt-6-sol'
RETRY_STATUSES = {408, 429, 500, 502, 503, 504}
GATEWAY_URL = 'http://127.0.0.1:8769'


def install_transport(out, plan, credential_file):
    paths = ('pm3_generator_harness.py', 'pm3_mechanism_experience.py',
        'source_snapshot/src/ohpu/llm.py', 'source_snapshot/src/ohpu/per_user_adaptation/phase_a.py')
    patches = []
    for relative in paths:
        path = out / relative
        source = path.read_text()
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        if relative.endswith('/llm.py'):
            old = '    parsed = urlsplit(endpoint)\n'
            new = ('    if (azure_endpoint == "http://127.0.0.1:8769"\n'
                '            and os.environ.get("OHPU_AZURE_FAILOVER_URL") == azure_endpoint):\n'
                '        if expected is not None and expected.rstrip("/") != azure_endpoint:\n'
                '            raise ValueError("Failover destination differs from run authorization")\n'
                '        return azure_endpoint\n' + old)
        elif relative.endswith('/phase_a.py'):
            old = '        for key, value in provider.items():\n'
            new = ('        if endpoint == "http://127.0.0.1:8769":\n'
                '            provider.update({"model_providers.azure.request_max_retries": "0",\n'
                '                "model_providers.azure.stream_max_retries": "0",\n'
                '                "model_providers.azure.stream_idle_timeout_ms": "930000"})\n' + old)
        else:
            old = '    if "swedencentral" not in os.environ.get("AZURE_OPENAI_ENDPOINT", ""):\n'
            new = ('    os.environ.pop("OHPU_AZURE_FAILOVER_URL", None)\n'
                '    if configuration.get("failover_url") == "http://127.0.0.1:8769":\n'
                '        if os.environ.get("AZURE_OPENAI_ENDPOINT") != configuration["failover_url"]:\n'
                '            raise ValueError("Failover credential endpoint mismatch")\n'
                '        os.environ["OHPU_AZURE_FAILOVER_URL"] = configuration["failover_url"]\n'
                '        return\n' + old)
        if source.count(old) != 1:
            raise ValueError('Unexpected failover anchor: ' + relative)
        source = source.replace(old, new)
        compile(source, relative, 'exec')
        path.write_text(source)
        patches.append({'file': relative, 'before_sha256': before,
            'after_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    for configuration in (plan['phg']['generator'], plan.setdefault('optimizer', {})):
        configuration.update(credential_file=str(credential_file), failover_url=GATEWAY_URL)
    plan['phg']['generator'].update(transport_retries=0, request_timeout_seconds=930,
        connection_keepalive=False, fresh_client_on_retry=False,
        retries='Gateway retries up to six endpoint attempts within a 900s scheduling budget; no outer retries')
    plan['endpoint_failover'] = {'url': GATEWAY_URL, 'endpoints': list(HOSTS), 'patches': patches,
        'max_endpoint_attempts': 6, 'request_budget_seconds': 900,
        'upstream_timeout_seconds': 300, 'cooldown_seconds': 60,
        'partial_stream_replay': False, 'buffer_until_response_completed': True}
    return plan['endpoint_failover']


class EndpointPool:
    def __init__(self, endpoints, token, log, *, timeout=300, cooldown=60, client=None,
                 max_attempts=6, request_budget=900, clock=time.monotonic, wait=None, enabled_endpoints=None):
        if len(endpoints) != 3 or tuple(urlsplit(row['url']).hostname for row in endpoints) != HOSTS:
            raise ValueError('Exactly the three authorized endpoints are required in registered order')
        for endpoint in endpoints:
            parsed = urlsplit(endpoint['url'])
            if (parsed.scheme != 'https' or parsed.path not in ('', '/') or parsed.port is not None
                    or parsed.username or parsed.password or parsed.query or parsed.fragment or not endpoint['key']):
                raise ValueError('Invalid endpoint configuration')
        if len(token) < 32:
            raise ValueError('A strong loopback authentication token is required')
        self.endpoints = endpoints
        self.token = token
        self.log = Path(log)
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.preferred = 0
        enabled = set(HOSTS if enabled_endpoints is None else enabled_endpoints)
        if not enabled or not enabled <= set(HOSTS):
            raise ValueError('Enabled endpoints must be a nonempty subset of authorized hosts')
        self.disabled = {index for index, host in enumerate(HOSTS) if host not in enabled}
        self.blocked_until = [0.0] * len(endpoints)
        self.cooldown = cooldown
        self.max_attempts = max_attempts
        self.request_budget = request_budget
        self.timeout = timeout
        self.clock = clock
        self.wait = wait or threading.Event().wait
        self.client = client or httpx.Client(timeout=httpx.Timeout(timeout, connect=15), follow_redirects=False)

    def order(self):
        with self.lock:
                return [index for offset in range(3)
                    if (index := (self.preferred + offset) % 3) not in self.disabled
                    and self.blocked_until[index] <= self.clock()]

    def attempts(self, call_id):
        deadline = self.clock() + self.request_budget
        attempt = 0
        while attempt < self.max_attempts and self.clock() < deadline:
            available = self.order()
            if not available:
                with self.lock:
                    eligible = [until for index, until in enumerate(self.blocked_until) if index not in self.disabled]
                    if not eligible:
                        return
                    delay = max(0.01, min(eligible) - self.clock())
                if delay >= deadline - self.clock():
                    return
                self.record(call_id=call_id, outcome='cooldown_wait', seconds=round(delay, 3))
                self.wait(delay)
                continue
            attempt += 1
            yield attempt, available[0], min(self.timeout, deadline - self.clock())

    def failed(self, index, retry_after=0):
        with self.lock:
            self.blocked_until[index] = self.clock() + max(self.cooldown, retry_after)
            if self.preferred == index:
                self.preferred = (index + 1) % 3

    def succeeded(self, index):
        with self.lock:
            self.preferred = index

    def missing_deployment(self, index):
        with self.lock:
            self.disabled.add(index)

    def record(self, **record):
        record['utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        with self.lock, self.log.open('a') as stream:
            stream.write(json.dumps(record) + '\n')


def retry_after(response):
    try:
        return min(3600, max(0, float(response.headers.get('retry-after', '0'))))
    except ValueError:
        return 0


def transient(response):
    if response.status_code in RETRY_STATUSES:
        return True
    if response.status_code == 404:
        try:
            return response.json().get('error', {}).get('code') == 'DeploymentNotFound'
        except (ValueError, AttributeError):
            return False
    return False


def error_metadata(response):
    try:
        error = response.json().get('error', {})
        return {key: value for key in ('code', 'type', 'param')
                if isinstance(value := error.get(key), str) and re.fullmatch(r'[\w.\[\]-]{1,100}', value)}
    except (ValueError, AttributeError):
        return {}


def completed_body(response, streaming, deadline):
    chunks = []
    size = 0
    for chunk in response.iter_bytes():
        size += len(chunk)
        if size > 16_000_000 or time.monotonic() > deadline:
            raise httpx.ReadTimeout('Buffered response exceeded size or time budget')
        chunks.append(chunk)
    body = b''.join(chunks)
    if streaming:
        events = []
        for line in body.splitlines():
            if line.startswith(b'data:') and line[5:].strip() != b'[DONE]':
                try:
                    events.append(json.loads(line[5:]))
                except ValueError:
                    raise httpx.RemoteProtocolError('Invalid event in buffered response') from None
        if not any(isinstance(event, dict) and event.get('type') == 'response.completed' for event in events):
            raise httpx.RemoteProtocolError('Stream ended before response.completed')
    else:
        try:
            json.loads(body)
        except ValueError:
            raise httpx.RemoteProtocolError('Incomplete JSON response') from None
    return body


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, format, *args):
        pass

    def reply(self, status, message):
        body = json.dumps({'error': {'message': message, 'type': 'endpoint_failover'}}).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Connection', 'close')
        if status == 503:
            self.send_header('Retry-After', '60')
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def do_GET(self):
        if self.path != '/health':
            return self.reply(404, 'Not found')
        return self.reply(200, 'ready')

    def do_POST(self):
        pool = self.server.pool
        if not hmac.compare_digest(self.headers.get('api-key', ''), pool.token):
            return self.reply(401, 'Invalid loopback credential')
        path = urlsplit(self.path)
        allowed = {'/openai/responses', '/openai/v1/responses',
            '/openai/deployments/' + MODEL + '/chat/completions'}
        if path.scheme or path.netloc or path.path not in allowed:
            return self.reply(400, 'Unregistered API path')
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 4_000_000 or self.headers.get('Transfer-Encoding'):
                raise ValueError('Invalid length')
            body = self.rfile.read(length)
            payload = json.loads(body)
            if not isinstance(payload, dict) or payload.get('model') != MODEL:
                raise ValueError('Unregistered model')
        except (ValueError, TypeError):
            return self.reply(400, 'Invalid registered request')
        call_id = uuid.uuid4().hex
        for attempt, index, remaining in pool.attempts(call_id):
            endpoint = pool.endpoints[index]
            started = time.monotonic()
            sent = False
            details = {'call_id': call_id, 'attempt': attempt, 'endpoint': HOSTS[index],
                'model': MODEL, 'path': path.path, 'stream': bool(payload.get('stream'))}
            try:
                with pool.client.stream('POST', endpoint['url'].rstrip('/') + self.path,
                        headers={'api-key': endpoint['key'], 'Content-Type': 'application/json',
                            'Accept': self.headers.get('Accept', '*/*')}, content=body,
                        timeout=httpx.Timeout(remaining, connect=min(15, remaining))) as response:
                    details['status'] = response.status_code
                    if response.status_code >= 400:
                        response.read()
                        details['upstream_error'] = error_metadata(response)
                        if response.status_code == 404 and details['upstream_error'].get('code') == 'DeploymentNotFound':
                            pool.missing_deployment(index)
                            details.update(outcome='deployment_unavailable', model_disabled=True)
                            continue
                        if transient(response):
                            pool.failed(index, retry_after(response))
                            details.update(outcome='retryable_status', usage_unknown=True)
                            continue
                        details.update(outcome='nonretryable_status', usage_unknown=True)
                        return self.reply(response.status_code, 'Upstream rejected request; inspect endpoint status')
                    if response.status_code != 200:
                        details['outcome'] = 'unexpected_status'
                        return self.reply(502, 'Unexpected upstream status')
                    response_body = completed_body(response, bool(payload.get('stream')), started + remaining)
                    self.send_response(200)
                    self.send_header('Content-Type', response.headers.get('content-type', 'application/json'))
                    self.send_header('Content-Length', str(len(response_body)))
                    self.send_header('Connection', 'close')
                    self.end_headers()
                    sent = True
                    self.close_connection = True
                    self.wfile.write(response_body)
                    self.wfile.flush()
                    pool.succeeded(index)
                    details['outcome'] = 'completed'
                    return
            except httpx.TransportError as error:
                pool.failed(index)
                details.update(outcome='partial_response' if sent else 'transport_error',
                    error_type=type(error).__name__, usage_unknown=True)
                if sent:
                    self.close_connection = True
                    return
            except (BrokenPipeError, ConnectionResetError):
                details.update(outcome='client_disconnected', usage_unknown=True)
                self.close_connection = True
                return
            finally:
                pool.record(**details, seconds=round(time.monotonic() - started, 3))
        self.reply(503, 'All registered endpoints failed or are cooling down')


def serve(config, port, log):
    if config.stat().st_mode & 0o077:
        raise ValueError('Credential configuration must be owner-only')
    settings = json.loads(config.read_text())
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.daemon_threads = True
    server.pool = EndpointPool(settings['endpoints'], settings['token'], log,
        enabled_endpoints=settings.get('enabled_endpoints_by_model', {}).get(MODEL))
    print(json.dumps({'listening': '127.0.0.1', 'port': port, 'model': MODEL}), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        server.pool.client.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8769)
    parser.add_argument('--log', type=Path, required=True)
    options = parser.parse_args()
    os.umask(0o077)
    serve(options.config, options.port, options.log)