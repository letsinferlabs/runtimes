#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Stable Engine protocol frontend shared by one engine image.

The engine-specific executable supplies native launch and exact-token-count
functions. This module owns the authenticated TLS boundary, bounded proxying,
health, normalized telemetry, child lifecycle, and protocol verification.
"""

from __future__ import annotations

import argparse
import http.client
import http.server
import ipaddress
import json
import os
import pathlib
import re
import secrets
import signal
import socket
import ssl
import stat
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any


PROTOCOL_VERSION = 2
MAX_REQUEST_BYTES = 32 * 1024 * 1024
MAX_CONTROL_RESPONSE_BYTES = 64 * 1024 * 1024
HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
ARTIFACT_REFERENCE = re.compile(r"\$\{artifact:([a-z][a-z0-9._-]{0,62})\}")


class AdapterError(RuntimeError):
    """The image, runtime contract, or native engine is invalid."""


def _load_object(path: pathlib.Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise AdapterError(f"runtime config is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AdapterError(f"cannot read runtime config: {error}") from error
    if not isinstance(value, dict):
        raise AdapterError("runtime config must be an object")
    return value


def load_runtime(expected_engine: str) -> dict[str, Any]:
    if os.environ.get("LETSINFER_ENGINE_PROTOCOL") != str(PROTOCOL_VERSION):
        raise AdapterError("unsupported or missing LETSINFER_ENGINE_PROTOCOL")
    path = pathlib.Path(
        os.environ.get(
            "LETSINFER_RUNTIME_CONFIG",
            "/opt/letsinfer/runtime-pack/runtime.json",
        )
    )
    runtime = _load_object(path)
    engine = runtime.get("engine")
    if (
        runtime.get("schema_version") != 4
        or not isinstance(engine, dict)
        or engine.get("id") != expected_engine
        or engine.get("protocol") != {"version": PROTOCOL_VERSION}
    ):
        raise AdapterError("runtime does not match this Engine OCI protocol identity")
    logical_model = runtime.get("logical_model")
    artifacts = runtime.get("artifacts")
    serving = runtime.get("serving")
    if (
        not isinstance(logical_model, str)
        or not logical_model
        or not isinstance(artifacts, list)
        or not artifacts
        or not isinstance(serving, dict)
    ):
        raise AdapterError("runtime model/artifact/serving contract is invalid")
    return runtime


def artifact_path(runtime: Mapping[str, Any], name: str) -> str:
    matches = [
        value
        for value in runtime["artifacts"]
        if isinstance(value, dict) and value.get("name") == name
    ]
    if len(matches) != 1:
        raise AdapterError(f"runtime references unknown artifact {name!r}")
    artifact = matches[0]
    uri = artifact.get("uri")
    revision = artifact.get("revision")
    if (
        not isinstance(uri, str)
        or not uri.startswith("hf://")
        or not isinstance(revision, str)
        or re.fullmatch(r"[0-9a-f]{40}", revision) is None
    ):
        raise AdapterError(f"artifact {name!r} has an invalid immutable identity")
    repository = uri.removeprefix("hf://")
    if re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*",
        repository,
    ) is None:
        raise AdapterError(f"artifact {name!r} has an invalid repository")
    path = f"/models/{repository.replace('/', '--').lower()}/{revision}"
    if artifact.get("format") == "gguf-file":
        filename = artifact.get("filename")
        if not isinstance(filename, str) or "/" in filename or "\\" in filename:
            raise AdapterError(f"artifact {name!r} has an invalid filename")
        path += f"/{filename}"
    return path


def expand_arguments(runtime: Mapping[str, Any]) -> list[str]:
    raw = runtime["engine"].get("arguments")
    if not isinstance(raw, list) or any(
        not isinstance(value, str) or not value for value in raw
    ):
        raise AdapterError("runtime engine arguments must be a string array")
    result: list[str] = []
    for value in raw:
        match = ARTIFACT_REFERENCE.fullmatch(value)
        if match is not None:
            result.append(artifact_path(runtime, match.group(1)))
        elif "${artifact:" in value:
            raise AdapterError("artifact references must occupy a complete argument")
        else:
            result.append(value)
    return result


def child_environment(runtime: Mapping[str, Any]) -> dict[str, str]:
    supplied = runtime["engine"].get("environment")
    if not isinstance(supplied, dict):
        raise AdapterError("runtime engine environment must be an object")
    result = dict(os.environ)
    for name, value in supplied.items():
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None
            or name.startswith("LETSINFER_")
            or not isinstance(value, str)
        ):
            raise AdapterError("runtime engine environment is invalid")
        result[name] = value
    result.update({"HF_HUB_OFFLINE": "1", "HF_HOME": "/root/.cache/huggingface"})
    return result


def _secret(path_value: str) -> str:
    path = pathlib.Path(path_value)
    if path.is_symlink() or not path.is_file():
        raise AdapterError(f"secret is not a regular file: {path}")
    details = path.stat()
    if not stat.S_ISREG(details.st_mode) or stat.S_IMODE(details.st_mode) & 0o077:
        raise AdapterError(f"secret permissions are not private: {path}")
    value = path.read_text(encoding="ascii").strip()
    if len(value) < 32 or any(character.isspace() for character in value):
        raise AdapterError(f"secret is invalid: {path}")
    return value


def _port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((host, 0))
        return int(listener.getsockname()[1])


def backend_json(
    host: str,
    port: int,
    method: str,
    path: str,
    body: bytes | None = None,
) -> tuple[int, dict[str, Any]]:
    connection = http.client.HTTPConnection(host, port, timeout=30)
    try:
        headers = {"Connection": "close", "Accept": "application/json"}
        if body is not None:
            headers.update(
                {
                    "Content-Type": "application/json",
                    "Content-Length": str(len(body)),
                }
            )
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        payload = response.read(MAX_CONTROL_RESPONSE_BYTES + 1)
        if len(payload) > MAX_CONTROL_RESPONSE_BYTES:
            raise AdapterError("native engine control response is too large")
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AdapterError("native engine returned invalid JSON") from error
        if not isinstance(value, dict):
            raise AdapterError("native engine returned a non-object response")
        return response.status, value
    finally:
        connection.close()


class EngineServer(http.server.ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        *,
        api_key: str,
        backend_host: str,
        backend_port: int,
        model: str,
        engine: str,
        max_connections: int,
        count_tokens: Callable[[str, int, bytes, str], int],
    ) -> None:
        self.api_key = api_key
        self.backend_host = backend_host
        self.backend_port = backend_port
        self.model = model
        self.engine = engine
        self.connection_slots = threading.BoundedSemaphore(max_connections)
        self.request_queue_size = max_connections
        self.count_tokens = count_tokens
        self.tls_context: ssl.SSLContext | None = None
        self.active = 0
        self.completed = 0
        self.metrics_lock = threading.Lock()
        super().__init__(address, EngineHandler)

    def get_request(self) -> tuple[socket.socket, Any]:
        request, address = self.socket.accept()
        request.settimeout(30)
        if self.tls_context is not None:
            try:
                request = self.tls_context.wrap_socket(request, server_side=True)
            except BaseException:
                request.close()
                raise
        request.settimeout(None)
        return request, address


class EngineHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "Let's Infer Engine"
    sys_version = ""

    @property
    def engine_server(self) -> EngineServer:
        return self.server  # type: ignore[return-value]

    def log_message(self, _format: str, *_arguments: Any) -> None:
        return

    def _json(self, status: int, value: Mapping[str, Any]) -> None:
        body = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def _authorized(self) -> bool:
        scheme, separator, token = self.headers.get("Authorization", "").partition(" ")
        return bool(
            separator
            and scheme.lower() == "bearer"
            and secrets.compare_digest(token, self.engine_server.api_key)
        )

    def _body(self) -> bytes | None:
        if self.headers.get("Transfer-Encoding"):
            self._json(501, {"error": {"message": "chunked requests unsupported"}})
            return None
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._json(400, {"error": {"message": "invalid content length"}})
            return None
        if size < 0 or size > MAX_REQUEST_BYTES:
            self._json(413, {"error": {"message": "request body too large"}})
            return None
        return self.rfile.read(size)

    def _healthy(self) -> bool:
        try:
            status, value = backend_json(
                self.engine_server.backend_host,
                self.engine_server.backend_port,
                "GET",
                "/v1/models",
            )
            entries = value.get("data", value.get("models"))
            if not isinstance(entries, list):
                return False
            identifiers = {
                item.get("id")
                for item in entries
                if isinstance(item, dict)
                and isinstance(item.get("id"), str)
            }
            return status == 200 and self.engine_server.model in identifiers
        except (AdapterError, OSError, http.client.HTTPException):
            return False

    def _health(self) -> None:
        healthy = self._healthy()
        self._json(200 if healthy else 503, {"status": "ok" if healthy else "starting"})

    def _telemetry(self) -> None:
        with self.engine_server.metrics_lock:
            active = self.engine_server.active
            completed = self.engine_server.completed
        self._json(
            200,
            {
                "object": "engine_telemetry",
                "protocol": PROTOCOL_VERSION,
                "engine": self.engine_server.engine,
                "model": self.engine_server.model,
                "state": "ready" if self._healthy() else "starting",
                "requests": {"active": active, "completed": completed, "queued": None},
                "tokens": {"decode_per_second": None, "prefill_per_second": None},
                "cache": {"prefix_hit_rate": None, "kv_used_bytes": None},
                "timestamp_unix_ns": time.time_ns(),
            },
        )

    def _token_count(self) -> None:
        body = self._body()
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
        except (AdapterError, OSError, ValueError, http.client.HTTPException) as error:
            self._json(400, {"error": {"message": str(error), "code": "exact_count_failed"}})
            return
        self._json(
            200,
            {
                "object": "token_count",
                "model": self.engine_server.model,
                "prompt_tokens": count,
            },
        )

    def _proxy(self) -> None:
        if not self.path.startswith("/") or "://" in self.path:
            self._json(400, {"error": {"message": "invalid request target"}})
            return
        body = self._body() if self.command in {"POST", "PUT", "PATCH"} else None
        if self.command in {"POST", "PUT", "PATCH"} and body is None:
            return
        self.engine_server.connection_slots.acquire()
        with self.engine_server.metrics_lock:
            self.engine_server.active += 1
        connection = http.client.HTTPConnection(
            self.engine_server.backend_host,
            self.engine_server.backend_port,
            timeout=None,
        )
        headers_sent = False
        try:
            headers = {
                key: value
                for key, value in self.headers.items()
                if key.lower()
                not in HOP_HEADERS | {"authorization", "host", "content-length"}
            }
            headers["Host"] = f"{self.engine_server.backend_host}:{self.engine_server.backend_port}"
            headers["Connection"] = "close"
            if body is not None:
                headers["Content-Length"] = str(len(body))
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
                chunk = response.read1(64 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
            self.close_connection = True
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except (OSError, http.client.HTTPException):
            if not headers_sent:
                self._json(502, {"error": {"message": "native engine unavailable"}})
            else:
                self.close_connection = True
        finally:
            connection.close()
            with self.engine_server.metrics_lock:
                self.engine_server.active -= 1
                self.engine_server.completed += 1
            self.engine_server.connection_slots.release()

    def _dispatch(self) -> None:
        if self.path == "/health" and self.command == "GET":
            self._health()
            return
        if not self._authorized():
            self._json(401, {"error": {"message": "unauthorized"}})
            return
        if self.path == "/v1/letsinfer/telemetry" and self.command == "GET":
            self._telemetry()
        elif self.path == "/v1/letsinfer/token-count" and self.command == "POST":
            self._token_count()
        else:
            self._proxy()

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch()

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch()

    def do_DELETE(self) -> None:  # noqa: N802
        self._dispatch()

    def do_PUT(self) -> None:  # noqa: N802
        self._dispatch()


def run(
    expected_engine: str,
    build_command: Callable[[Mapping[str, Any], int], Sequence[str]],
    build_environment: Callable[[Mapping[str, Any]], Mapping[str, str]],
    count_tokens: Callable[[str, int, bytes, str], int],
    arguments: Sequence[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(prog="engine-adapter")
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--protocol", required=True, type=int)
    commands.add_parser("serve")
    options = parser.parse_args(arguments)
    if options.command == "verify":
        if options.protocol != PROTOCOL_VERSION:
            raise AdapterError("unsupported Engine protocol")
        print(json.dumps({"engine_protocol": PROTOCOL_VERSION, "status": "ok"}, separators=(",", ":")))
        return 0

    runtime = load_runtime(expected_engine)
    host = os.environ.get("LETSINFER_LISTEN_HOST", "127.0.0.1")
    backend_host = "127.0.0.1"
    try:
        if not ipaddress.ip_address(host).is_loopback:
            raise AdapterError("Engine protocol listen host must be loopback")
        port = int(os.environ["LETSINFER_LISTEN_PORT"])
    except (KeyError, ValueError) as error:
        raise AdapterError("Engine protocol listen address is invalid") from error
    if port not in range(1, 65536):
        raise AdapterError("Engine protocol listen port is invalid")
    backend_port = _port(backend_host)
    command = list(build_command(runtime, backend_port))
    if not command or any(not isinstance(value, str) or not value for value in command):
        raise AdapterError("native engine command is invalid")
    environment = child_environment(runtime)
    environment.update(build_environment(runtime))
    api_key = _secret(os.environ["LETSINFER_API_KEY_FILE"])
    server = EngineServer(
        (host, port),
        api_key=api_key,
        backend_host=backend_host,
        backend_port=backend_port,
        model=runtime["logical_model"],
        engine=expected_engine,
        max_connections=int(runtime["serving"]["max_connections"]),
        count_tokens=count_tokens,
    )
    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tls.minimum_version = ssl.TLSVersion.TLSv1_2
    tls.load_cert_chain(
        os.environ["LETSINFER_TLS_CERT_FILE"],
        os.environ["LETSINFER_TLS_KEY_FILE"],
    )
    server.tls_context = tls
    stopping = threading.Event()

    def stop(_signum: int, _frame: Any) -> None:
        stopping.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop)
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
            try:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=110)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
