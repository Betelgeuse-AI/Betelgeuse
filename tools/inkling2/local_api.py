"""Loopback Chat Completions adapter for the standalone bel-inkling2 engine."""

import argparse
import atexit
import json
import math
import os
import subprocess
import threading
import time
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from jinja2 import Environment, TemplateError
from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream

MODEL = "Inkling-2-Mini"
MAX_REQUEST_BYTES = 4 * 1024 * 1024
DEFAULTS = {"temperature": 0.6, "top_p": 0.95, "top_k": 0, "min_p": 0.0}


def normalize_request(request):
    messages = request["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError("Provide at least one chat message.")
    for item in messages:
        role, content = item["role"], item.get("content")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Unsupported message role: {role}")
        if isinstance(content, list) and all(
            isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)
            for part in content
        ):
            content = "\n".join(part["text"] for part in content)
        if content is None and role == "assistant":
            content = ""
        if not isinstance(content, str):
            raise ValueError("Only text message content is supported.")
        item["content"] = content
        if role == "tool" and not isinstance(item.get("tool_call_id"), str):
            raise ValueError("Tool messages require a tool_call_id.")
        for call in item.get("tool_calls") or []:
            if role != "assistant" or call.get("type") != "function":
                raise ValueError("Only assistant function calls are supported.")
            function = call["function"]
            arguments = function["arguments"]
            function["arguments"] = json.loads(arguments) if isinstance(arguments, str) else arguments
            if not isinstance(function["arguments"], dict):
                raise ValueError("Function arguments must be a JSON object.")
    tools = request.get("tools") or []
    if not isinstance(tools, list) or any(
        tool.get("type") != "function" or not isinstance(tool["function"]["name"], str) for tool in tools
    ):
        raise ValueError("Only function tools are supported.")
    choice = request.get("tool_choice")
    if choice == "none":
        tools = []
    elif choice not in (None, "auto"):
        raise ValueError("Supported tool_choice values are auto and none.")
    return messages, tools


class CompletionDecoder:
    def __init__(self, tokenizer, names, parallel):
        self.tokenizer = tokenizer
        self.markers = {tokenizer.token_to_id(f"<|{name}|>"): name for name in (
            "message_model", "content_text", "content_thinking", "content_invoke_tool_json",
            "end_message", "content_model_end_sampling",
        )}
        self.names = names
        self.parallel = parallel
        self.channel = "content"
        self.decoder = DecodeStream(skip_special_tokens=True)
        self.payload = []
        self.calls = 0

    def finish_call(self):
        try:
            data = json.loads("".join(self.payload))
            name, args = data["name"], data["args"]
            if not isinstance(name, str) or name not in self.names or not isinstance(args, dict):
                return None
            arguments = json.dumps(args, ensure_ascii=False, allow_nan=False)
        except (KeyError, TypeError, ValueError):
            return None
        if self.calls and not self.parallel:
            return None
        call = {"index": self.calls, "id": "call_" + uuid.uuid4().hex, "type": "function",
                "function": {"name": name, "arguments": arguments}}
        self.calls += 1
        return {"tool_calls": [call]}

    def step(self, token):
        if marker := self.markers.get(token):
            delta = self.finish_call() if marker == "end_message" and self.channel == "tool" else None
            self.channel = {
                "content_text": "content", "content_thinking": "reasoning_content",
                "content_invoke_tool_json": "tool", "message_model": "header",
            }.get(marker, "content")
            self.payload.clear()
            self.decoder = DecodeStream(skip_special_tokens=True)
            return delta
        piece = self.decoder.step(self.tokenizer, token)
        if piece and self.channel == "tool":
            self.payload.append(piece)
        if piece and self.channel in {"content", "reasoning_content"}:
            return {self.channel: piece}
        return None


class Engine:
    def __init__(self, args):
        self.checkpoint = args.model.resolve()
        self.tokenizer = Tokenizer.from_file(str(self.checkpoint / "tokenizer.json"))
        template = (self.checkpoint / "chat_template.jinja").read_text(encoding="utf-8")
        env = Environment()
        env.globals["raise_exception"] = self.invalid_template
        env.filters["tojson"] = lambda value, **options: json.dumps(value, ensure_ascii=False, **options)
        self.template = env.from_string(
            template.replace("{%- generation -%}", "").replace("{%- endgeneration -%}", "")
        )
        self.lock = threading.Lock()
        self.input_lock = threading.Lock()
        self.active = threading.Event()
        self.log = args.log.open("w", encoding="utf-8")
        config = {
            "INK2_GPU_FETCH": "0",
            "INK2_PIN_HOST_MIB": "512",
            "INK2_CPU_ACT": "int8",
            "INK2_CPU_OPT": "29",
            "INK2_RING_EXTRA": "0",
        }
        env_vars = {**config, **os.environ}
        if cuda_path := env_vars.get("CUDA_PATH"):
            env_vars["PATH"] = str(Path(cuda_path) / "bin") + os.pathsep + env_vars["PATH"]
        self.process = subprocess.Popen(
            [str(args.engine.resolve()),
             "--model", str(self.checkpoint), "--serve",
             "--max-context", str(args.context),
             "--chunk", str(args.chunk),
             "--graph", "1", "--tc", "1", "--kv", "k8v4",
             "--vram-reserve-mib", "0", "--cpu-threads", "5", "--kv-grow", "1",
             "--kv-grow-step", "4096", "--spec", "0"],
            env=env_vars, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        atexit.register(self.close)
        for line in self.process.stdout:
            if line.startswith("READY "):
                self.context = int(line.split()[1])
                return
            print(line.rstrip(), flush=True)
        raise RuntimeError(f"Betelgeuse failed to start. See {args.log}.")

    @staticmethod
    def invalid_template(message):
        raise ValueError(message)

    def encode(self, messages, tools, effort):
        prompt = self.template.render(messages=messages, tools=tools, reasoning_effort=effort)
        ids = self.tokenizer.encode(prompt, add_special_tokens=False).ids
        if len(ids) >= self.context:
            raise ValueError(f"The prompt needs {len(ids)} tokens. Context is {self.context}.")
        return ids

    def send(self, command):
        with self.input_lock:
            self.process.stdin.write(command + "\n")
            self.process.stdin.flush()

    def stop(self):
        if self.active.is_set() and self.process.poll() is None:
            self.send("STOP")

    def close(self):
        if self.process.poll() is None:
            try:
                self.send("QUIT")
                self.process.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                self.process.kill()
        self.log.close()

    def generate(self, ids, limit, options, decoder, emit):
        params = " ".join(f"{key}={value}" for key, value in options.items())
        tokens = ",".join(map(str, ids))
        with self.lock:
            self.send(f"GEN {limit} {params} {tokens}")
            self.active.set()
            disconnected = False
            try:
                for line in self.process.stdout:
                    kind, _, payload = line.strip().partition(" ")
                    if kind == "T":
                        if disconnected:
                            continue
                        if delta := decoder.step(int(payload)):
                            try:
                                emit(delta)
                            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                                disconnected = True
                                self.stop()
                    elif kind == "PP" and len(ids) > 8192:
                        processed, total, _, rate = payload.split()
                        if int(processed) % 4096 == 0 or processed == total:
                            print(f"Prefill {processed}/{total} tokens at {rate} tokens/s", flush=True)
                    elif kind == "DONE":
                        generated, prompt, pp_ms, decode_ms, reason, _, _, cached, _, ttft = payload.split()
                        return int(generated), int(prompt), float(pp_ms), float(decode_ms), reason, int(cached), float(ttft)
                    elif kind == "ERR":
                        raise RuntimeError(payload)
                raise RuntimeError(f"Betelgeuse exited. See {self.log.name}.")
            finally:
                self.active.clear()


class ChatHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        server = args[2]
        super().__init__(*args, directory=server.ui, **kwargs)

    def write_json(self, body, status=200):
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def is_loopback_request(self):
        port = self.server.server_port
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        origins = {None, *(f"http://{host}" for host in hosts)}
        return self.headers.get("Host") in hosts and self.headers.get("Origin") in origins

    def do_HEAD(self):
        if not self.is_loopback_request():
            return self.send_error(403)
        if self.server.ui is None:
            return self.send_error(404)
        return super().do_HEAD()

    def do_GET(self):
        if not self.is_loopback_request():
            return self.write_json({"error": {"message": "Loopback requests only."}}, 403)
        route = urlsplit(self.path).path
        if route == "/health":
            if self.server.engine.process.poll() is not None:
                return self.write_json({"status": "engine exited"}, 503)
            return self.write_json({"status": "ok"})
        if route in ("/models", "/v1/models"):
            return self.write_json({"object": "list", "data": [
                {"id": MODEL, "object": "model", "owned_by": "Betelgeuse", "created": 0}
            ]})
        if route == "/props":
            return self.write_json({
                "model_alias": MODEL, "model_path": str(self.server.engine.checkpoint), "total_slots": 1,
                "default_generation_settings": {"n_ctx": self.server.engine.context, "params": DEFAULTS},
                "modalities": {"vision": False, "audio": False},
                "ui_settings": {}, "cors_proxy_enabled": False,
            })
        if route == "/tools":
            return self.write_json([])
        if self.server.ui is None:
            return self.write_json({"error": {"message": "Unknown route."}}, 404)
        return super().do_GET()

    def do_POST(self):
        if not self.is_loopback_request():
            return self.write_json({"error": {"message": "Loopback requests only."}}, 403)
        route = urlsplit(self.path).path
        if route not in ("/v1/chat/completions", "/v1/streams/lookup", "/tools"):
            return self.write_json({"error": {"message": "Unknown route."}}, 404)
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= MAX_REQUEST_BYTES:
                raise ValueError("The request body must be between 1 byte and 4 MiB.")
            request = json.loads(self.rfile.read(length))
            if route == "/v1/streams/lookup":
                return self.write_json([])
            if route == "/tools":
                return self.write_json({"error": "Local tools are disabled."}, 403)
            if request.get("model", MODEL) != MODEL:
                return self.write_json({"error": {"message": "Model not found."}}, 404)
            messages, tools = normalize_request(request)
            engine = self.server.engine
            ids = engine.encode(messages, tools, request.get("reasoning_effort", "none"))
            decoder = CompletionDecoder(engine.tokenizer, {tool["function"]["name"] for tool in tools},
                                        request.get("parallel_tool_calls", True))
            limit = int(request.get("max_completion_tokens") or request.get("max_tokens") or 512)
            limit = min(max(1, limit), engine.context - len(ids))
            options = {key: float(request.get(key, value)) for key, value in DEFAULTS.items()}
            if not all(map(math.isfinite, options.values())):
                raise ValueError("Sampling parameters must be finite.")
            options["top_k"] = int(options["top_k"])
            options["cache_prompt"] = int(request.get("cache_prompt") is not False)
            if request.get("seed") is not None and int(request["seed"]) >= 0:
                options["seed"] = int(request["seed"])
        except (AttributeError, KeyError, OverflowError, TypeError, ValueError, TemplateError) as error:
            return self.write_json({"error": {"message": str(error)}}, 400)
        try:
            self.complete(request.get("stream", False), ids, limit, options, decoder)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def complete(self, stream, ids, limit, options, decoder):
        base = {"id": "chatcmpl-" + uuid.uuid4().hex, "created": int(time.time()), "model": MODEL}
        message = {"role": "assistant", "content": ""}

        def event(body):
            data = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)
            self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
            self.wfile.flush()

        def chunk(delta, finish=None, **extra):
            event({**base, "object": "chat.completion.chunk",
                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **extra})

        def collect(delta):
            for key, value in delta.items():
                if key == "tool_calls":
                    message.setdefault(key, []).extend({k: v for k, v in call.items() if k != "index"} for call in value)
                else:
                    message[key] = message.get(key, "") + value

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            chunk({"role": "assistant"})
        try:
            generated, prompt, pp_ms, decode_ms, finish, cached, ttft = self.server.engine.generate(
                ids, limit, options, decoder, chunk if stream else collect
            )
        except (OSError, RuntimeError) as error:
            body = {"error": {"message": str(error)}}
            if stream:
                event(body)
                return event("[DONE]")
            return self.write_json(body, 503)
        usage = {"prompt_tokens": prompt, "completion_tokens": generated,
                 "total_tokens": prompt + generated, "prompt_tokens_details": {"cached_tokens": cached}}
        timings = {
            "prompt_n": prompt - cached, "prompt_ms": pp_ms,
            "prompt_per_second": (prompt - cached) * 1000 / max(pp_ms, 1),
            "predicted_n": generated, "predicted_ms": decode_ms,
            "predicted_per_second": generated * 1000 / max(decode_ms, 1), "ttft_ms": ttft,
        }
        finish = "tool_calls" if decoder.calls else ("length" if finish == "length" else "stop")
        if stream:
            chunk({}, finish, usage=usage, timings=timings)
            return event("[DONE]")
        if decoder.calls and not message["content"]:
            message["content"] = None
        return self.write_json({
            **base, "object": "chat.completion",
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": usage, "timings": timings,
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True, help="Path to bel-inkling2")
    parser.add_argument("--model", type=Path, required=True, help="Original checkpoint directory")
    parser.add_argument("--ui", type=Path, help="Optional built Betelgeuse UI directory")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--log", type=Path, default=Path("inkling-engine.log"))
    args = parser.parse_args()
    with ThreadingHTTPServer(("127.0.0.1", args.port), ChatHandler) as server:
        server.ui = str(args.ui.resolve()) if args.ui else None
        server.engine = Engine(args)
        print(f"Betelgeuse is serving http://127.0.0.1:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
