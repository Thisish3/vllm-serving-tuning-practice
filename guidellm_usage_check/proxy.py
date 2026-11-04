"""HTTP proxy that hides token usage from the client but logs it.

    PROXY_LOG=log.jsonl python proxy.py [listen_port] [upstream]

For POST bodies it removes ignore_eos and min_tokens, so generation can stop at
EOS, and forces stream_options.include_usage so vLLM still reports usage. For
streamed responses it records completion_tokens from the usage event, deletes
the usage field, and forwards the event. One JSON line is logged per completion.
"""

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
UPSTREAM = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000"
LOG = os.environ.get("PROXY_LOG", "proxy_log.jsonl")
HOP = {"host", "content-length", "connection", "transfer-encoding", "accept-encoding"}

client = httpx.Client(base_url=UPSTREAM, timeout=None)
lock = threading.Lock()
seq = 0


def log(entry):
    global seq
    with lock:
        seq += 1
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"seq": seq, **entry}, ensure_ascii=False) + "\n")


def text_of(choice):
    delta = choice.get("delta") or choice.get("message") or {}
    return choice.get("text") or delta.get("content") or ""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def send_chunk(self, data):
        if not data:  # a zero-length chunk would end the chunked body early
            return
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def forward(self):
        length = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(length) if length else None
        body = None
        if raw and "json" in (self.headers.get("content-type") or ""):
            body = json.loads(raw)
            sent = {k: body.get(k) for k in ("ignore_eos", "min_tokens") if k in body}
            body.pop("ignore_eos", None)
            body.pop("min_tokens", None)
            if body.get("stream"):
                body.setdefault("stream_options", {})["include_usage"] = True
            raw = json.dumps(body).encode()
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        start = time.time()
        req = client.build_request(self.command, self.path, content=raw, headers=headers)
        resp = client.send(req, stream=True)
        try:
            self.send_response(resp.status_code)
            for k, v in resp.headers.items():
                if k.lower() not in HOP:
                    self.send_header(k, v)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            state = {"usage": None, "text": [], "finish": None}
            if body is not None and body.get("stream"):
                for line in resp.iter_lines():
                    self.send_chunk((self.filter_sse(line, state) + "\n").encode())
            else:
                data = resp.read()
                if body is not None and resp.status_code == 200:
                    obj = json.loads(data)
                    state["usage"] = obj.pop("usage", None)
                    for c in obj.get("choices", []):
                        state["text"].append(text_of(c))
                        state["finish"] = c.get("finish_reason")
                    data = json.dumps(obj).encode()
                self.send_chunk(data)
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        finally:
            resp.close()
        if body is not None:
            usage = state["usage"] or {}
            log({
                "path": self.path,
                "start": start,
                "end": time.time(),
                "status": resp.status_code,
                "removed": sent,
                "max_tokens": body.get("max_tokens", body.get("max_completion_tokens")),
                "completion_tokens": usage.get("completion_tokens"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "finish_reason": state["finish"],
                "text": "".join(state["text"]),
            })

    def filter_sse(self, line, state):
        if not line.startswith("data:") or line[5:].strip() == "[DONE]":
            return line
        obj = json.loads(line[5:])
        if obj.get("usage"):
            state["usage"] = obj["usage"]
        obj.pop("usage", None)
        for c in obj.get("choices", []):
            state["text"].append(text_of(c))
            state["finish"] = c.get("finish_reason") or state["finish"]
        return "data: " + json.dumps(obj, ensure_ascii=False)

    do_GET = do_POST = forward


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
