#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""TLS/authenticated Engine protocol frontend for a native loopback backend."""

from __future__ import annotations

import http.client
import http.server
import json
import os
import pathlib
import re
import secrets
import signal
import ssl
import stat
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any


PROTOCOL_VERSION = 2
MAX_BODY = 32 << 20
HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade",
}


class AdapterError(RuntimeError):
    pass


def load_runtime(expected_engine: str) -> dict[str, Any]:
    path = pathlib.Path(os.environ["LETSINFER_RUNTIME_CONFIG"])
    if path.is_symlink() or not path.is_file():
        raise AdapterError("runtime config is unavailable")
    value = json.loads(path.read_text(encoding="utf-8"))
    engine = value.get("engine") if isinstance(value, dict) else None
    if (
        value.get("schema_version") != 6
        or not isinstance(engine, dict)
        or engine.get("id") != expected_engine
        or engine.get("protocol") != {"version": PROTOCOL_VERSION}
    ):
        raise AdapterError("runtime does not match the native Engine")
    return value


def artifact_path(runtime: Mapping[str, Any], name: str) -> str:
    matches = [item for item in runtime["artifacts"] if item.get("name") == name]
    if len(matches) != 1:
        raise AdapterError("runtime model artifact is ambiguous")
    artifact = matches[0]
    repository = str(artifact["uri"]).removeprefix("hf://")
    revision = str(artifact["revision"])
    if re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*",
        repository,
    ) is None or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise AdapterError("runtime model artifact identity is invalid")
    root = pathlib.Path(os.environ["LETSINFER_MODEL_ROOT"])
    path = root / repository.replace("/", "--").lower() / revision
    if artifact.get("format") == "gguf-file":
        filename = artifact.get("filename")
        if not isinstance(filename, str) or "/" in filename or "\\" in filename:
            raise AdapterError("runtime GGUF filename is invalid")
        path /= filename
    return str(path)


def expand_arguments(runtime: Mapping[str, Any]) -> list[str]:
    values = runtime["engine"].get("arguments")
    if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
        raise AdapterError("runtime Engine arguments are invalid")
    result: list[str] = []
    for item in values:
        match = re.fullmatch(r"\$\{artifact:([a-z][a-z0-9._-]{0,62})\}", item)
        if match is not None:
            result.append(artifact_path(runtime, match.group(1)))
        elif "${artifact:" in item:
            raise AdapterError("artifact reference must occupy a complete argument")
        else:
            result.append(item)
    return result


def private_secret(path_value: str) -> str:
    path = pathlib.Path(path_value)
    if path.is_symlink() or not path.is_file():
        raise AdapterError("Engine credential is unavailable")
    details = path.stat()
    if not stat.S_ISREG(details.st_mode) or stat.S_IMODE(details.st_mode) & 0o077:
        raise AdapterError("Engine credential permissions are unsafe")
    value = path.read_text(encoding="ascii").strip()
    if len(value) < 32 or any(character.isspace() for character in value):
        raise AdapterError("Engine credential is invalid")
    return value


def backend_json(
    host: str,
    port: int,
    method: str,
    path: str,
    body: bytes | None = None,
) -> tuple[int, dict[str, Any]]:
    connection = http.client.HTTPConnection(host, port, timeout=30)
    try:
        connection.request(
            method,
            path,
            body=body,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        raw = response.read(MAX_BODY + 1)
    finally:
        connection.close()
    if len(raw) > MAX_BODY:
        raise AdapterError("native Engine response is too large")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AdapterError("native Engine response is invalid JSON") from error
    if not isinstance(value, dict):
        raise AdapterError("native Engine response must be an object")
    return response.status, value


class EngineServer(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        *,
        api_key: str,
        backend_port: int,
        engine: str,
        model: str,
        count_tokens: Callable[[str, int, bytes, str], int],
        max_active_requests: int,
    ) -> None:
        super().__init__(address, EngineHandler)
        self.api_key = api_key
        self.backend_host = "127.0.0.1"
        self.backend_port = backend_port
        self.engine = engine
        self.model = model
        self.count_tokens = count_tokens
        self.slots = threading.BoundedSemaphore(max_active_requests)
        self.lock = threading.Lock()
        self.active = 0
        self.completed = 0


class EngineHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "Let's Infer Native Engine"
    sys_version = ""

    @property
    def engine_server(self) -> EngineServer:
        return self.server  # type: ignore[return-value]

    def log_message(self, _format: str, *_arguments: Any) -> None:
        return

    def json_response(self, status: int, value: Mapping[str, Any]) -> None:
        body = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def authorized(self) -> bool:
        scheme, separator, token = self.headers.get("Authorization", "").partition(" ")
        return bool(separator and scheme.lower() == "bearer" and secrets.compare_digest(
            token, self.engine_server.api_key
        ))

    def body(self) -> bytes | None:
        if self.headers.get("Transfer-Encoding"):
            self.json_response(501, {"error": {"message": "chunked requests unsupported"}})
            return None
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.json_response(400, {"error": {"message": "invalid content length"}})
            return None
        if size < 0 or size > MAX_BODY:
            self.json_response(413, {"error": {"message": "request body too large"}})
            return None
        return self.rfile.read(size)

    def healthy(self) -> bool:
        try:
            status, value = backend_json(
                self.engine_server.backend_host,
                self.engine_server.backend_port,
                "GET",
                "/health",
            )
            return status == 200 and value.get("status") == "ok"
        except (AdapterError, OSError, http.client.HTTPException):
            return False

    def telemetry(self) -> None:
        with self.engine_server.lock:
            active = self.engine_server.active
            completed = self.engine_server.completed
        self.json_response(200, {
            "object": "engine_telemetry",
            "protocol": PROTOCOL_VERSION,
            "engine": self.engine_server.engine,
            "model": self.engine_server.model,
            "state": "ready" if self.healthy() else "starting",
            "requests": {"active": active, "completed": completed, "queued": None},
            "tokens": {"decode_per_second": None, "prefill_per_second": None},
            "cache": {"prefix_hit_rate": None, "kv_used_bytes": None},
            "timestamp_unix_ns": time.time_ns(),
        })

    def token_count(self) -> None:
        body = self.body()
        if body is None:
            return
        try:
            request = json.loads(body)
            if not isinstance(request, dict) or request.get("model") != self.engine_server.model:
                raise AdapterError("request model does not match runtime")
            count = self.engine_server.count_tokens(
                self.engine_server.backend_host,
                self.engine_server.backend_port,
                body,
                self.engine_server.model,
            )
            if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                raise AdapterError("native Engine token count is invalid")
        except (AdapterError, OSError, ValueError, http.client.HTTPException) as error:
            self.json_response(400, {"error": {"message": str(error), "code": "exact_count_failed"}})
            return
        self.json_response(200, {
            "object": "token_count",
            "model": self.engine_server.model,
            "prompt_tokens": count,
        })

    def proxy(self) -> None:
        if not self.path.startswith("/") or "://" in self.path:
            self.json_response(400, {"error": {"message": "invalid request target"}})
            return
        body = self.body() if self.command in {"POST", "PUT", "PATCH"} else None
        if self.command in {"POST", "PUT", "PATCH"} and body is None:
            return
        if body is not None and self.engine_server.engine == "mlx-lm":
            try:
                request = json.loads(body)
                if isinstance(request, dict) and "model" in request:
                    request["model"] = "default_model"
                    body = json.dumps(request, separators=(",", ":")).encode()
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass
        self.engine_server.slots.acquire()
        with self.engine_server.lock:
            self.engine_server.active += 1
        connection = http.client.HTTPConnection(
            self.engine_server.backend_host,
            self.engine_server.backend_port,
            timeout=None,
        )
        headers_sent = False
        try:
            headers = {
                key: value for key, value in self.headers.items()
                if key.lower() not in HOP_HEADERS | {"authorization", "host", "content-length"}
            }
            if body is not None:
                headers["Content-Length"] = str(len(body))
            headers["Connection"] = "close"
            connection.request(self.command, self.path, body=body, headers=headers)
            response = connection.getresponse()
            self.send_response(response.status, response.reason)
            for key, value in response.getheaders():
                if key.lower() not in HOP_HEADERS | {"date", "server"}:
                    self.send_header(key, value)
            self.send_header("Connection", "close")
            self.end_headers()
            headers_sent = True
            while True:
                chunk = response.read1(64 << 10)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
            self.close_connection = True
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except (OSError, http.client.HTTPException):
            if not headers_sent:
                self.json_response(502, {"error": {"message": "native Engine unavailable"}})
            else:
                self.close_connection = True
        finally:
            connection.close()
            with self.engine_server.lock:
                self.engine_server.active -= 1
                self.engine_server.completed += 1
            self.engine_server.slots.release()

    def dispatch(self) -> None:
        if self.path == "/health" and self.command == "GET":
            healthy = self.healthy()
            self.json_response(200 if healthy else 503, {
                "status": "ok" if healthy else "starting"
            })
            return
        if not self.authorized():
            self.json_response(401, {"error": {"message": "unauthorized"}})
            return
        if self.path == "/v1/models" and self.command == "GET":
            self.json_response(200, {
                "object": "list",
                "data": [{
                    "id": self.engine_server.model,
                    "object": "model",
                    "owned_by": "letsinfer",
                }],
            })
            return
        if self.path == "/v1/letsinfer/telemetry" and self.command == "GET":
            self.telemetry()
        elif self.path == "/v1/letsinfer/token-count" and self.command == "POST":
            self.token_count()
        else:
            self.proxy()

    do_GET = dispatch
    do_POST = dispatch
    do_DELETE = dispatch
    do_PUT = dispatch


def run(
    expected_engine: str,
    build_command: Callable[[Mapping[str, Any], int], Sequence[str]],
    count_tokens: Callable[[str, int, bytes, str], int],
    verify_engine: Callable[[], None] | None = None,
    arguments: Sequence[str] | None = None,
) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="engine-adapter")
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--protocol", required=True, type=int)
    commands.add_parser("serve")
    options = parser.parse_args(arguments)
    if options.command == "verify":
        if options.protocol != PROTOCOL_VERSION:
            raise AdapterError("unsupported Engine protocol")
        if verify_engine is not None:
            verify_engine()
        print(json.dumps({"engine_protocol": PROTOCOL_VERSION, "status": "ok"}, separators=(",", ":")))
        return 0

    runtime = load_runtime(expected_engine)
    if os.environ.get("LETSINFER_ENGINE_PROTOCOL") != str(PROTOCOL_VERSION):
        raise AdapterError("Engine protocol environment is invalid")
    host = os.environ.get("LETSINFER_LISTEN_HOST", "127.0.0.1")
    port = int(os.environ["LETSINFER_LISTEN_PORT"])
    backend_port = int(os.environ["LETSINFER_NATIVE_BACKEND_PORT"])
    if host not in {"127.0.0.1", "0.0.0.0"} or port == backend_port:
        raise AdapterError("native Engine port allocation is invalid")
    command = list(build_command(runtime, backend_port))
    environment = dict(os.environ)
    environment.update(runtime["engine"].get("environment", {}))
    api_key = private_secret(os.environ["LETSINFER_API_KEY_FILE"])
    server = EngineServer(
        (host, port),
        api_key=api_key,
        backend_port=backend_port,
        engine=expected_engine,
        model=runtime["logical_model"],
        count_tokens=count_tokens,
        max_active_requests=int(runtime["serving"]["max_active_requests"]),
    )
    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tls.minimum_version = ssl.TLSVersion.TLSv1_3
    tls.maximum_version = ssl.TLSVersion.TLSv1_3
    tls.load_cert_chain(
        os.environ["LETSINFER_TLS_CERT_FILE"],
        os.environ["LETSINFER_TLS_KEY_FILE"],
    )
    server.socket = tls.wrap_socket(server.socket, server_side=True)
    stopping = threading.Event()

    def stop(_signum: int, _frame: Any) -> None:
        stopping.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    child = subprocess.Popen(command, env=environment, start_new_session=True)
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    try:
        while child.poll() is None and not stopping.wait(0.25):
            pass
        return 0 if stopping.is_set() else int(child.returncode or 1)
    finally:
        server.shutdown()
        server.server_close()
        serving.join(timeout=2)
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
