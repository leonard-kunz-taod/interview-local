"""The local Ollama service, answering the real OllamaClient in-process."""

import contextlib
import io
import json
import urllib.error
import urllib.parse
import urllib.request
import urllib.response
from email.message import Message
from unittest.mock import patch

from interview_local.llm import EDIT_PROMPT, LOCAL_ENDPOINT, MODEL, REVIEW_PROMPT

DIGEST = "sha256-ministral"
QUANTIZATION = "Q4_K_M"
IDENTITY = {"model": MODEL, "digest": DIGEST, "quantization": QUANTIZATION}
LOCAL_HOST = urllib.parse.urlsplit(LOCAL_ENDPOINT).netloc
CHAT_ROLES = {"system", "user", "assistant", "tool"}
PROSE_REPLY = "Here is the edited transcript you asked for."
MOVED = "-moved"


class ScriptedModel:
    """The editor and reviewer model behind the service; tests script its answers."""

    def __init__(
        self,
        invalid_edits=0,
        wording=None,
        speaker="A",
        split=False,
        review_id_offset=0,
        paragraph_fields=None,
        review_fields=None,
        response=None,
    ):
        self.invalid_edits = invalid_edits
        self.wording = wording
        self.speaker = speaker
        self.split = split
        self.review_id_offset = review_id_offset
        self.paragraph_fields = paragraph_fields or {}
        self.review_fields = review_fields or {}
        self.response = response

    def answer(self, instruction, data):
        if self.response is not None:
            return self.response
        if instruction == EDIT_PROMPT:
            return self.edit(data["editable_words"])
        return self.review(data["paragraphs"])

    def edit(self, words):
        invalid = self.invalid_edits > 0
        self.invalid_edits -= 1
        middle = len(words) // 2
        groups = [words[:middle], words[middle:]] if self.split else [words]
        paragraphs = [self.paragraph(group) for group in groups]
        if invalid:
            paragraphs[0]["first_word"] += 1
        return {"paragraphs": paragraphs}

    def paragraph(self, words):
        return {
            "first_word": words[0]["id"],
            "last_word": words[-1]["id"],
            "speaker": self.speaker,
            "text": self.wording or " ".join(w["t"] for w in words),
            "uncertain": False,
            "reason": "Grouped into a readable paragraph.",
            **self.paragraph_fields,
        }

    def review(self, paragraphs):
        return {
            "reviews": [
                {
                    "paragraph_id": p["paragraph_id"] + self.review_id_offset,
                    "uncertain": False,
                    "reasons": [],
                    **self.review_fields,
                }
                for p in paragraphs
            ]
        }


class FakeOllama(urllib.request.BaseHandler):
    """Answers the local HTTP API the way Ollama does.

    Runs inside the client's real opener, after its proxy handling and before
    the network: a request routed to a proxy never reaches it. A path listed in
    redirects answers 302 to the same endpoint under a moved path.
    """

    handler_order = 499

    def __init__(
        self,
        model=None,
        installed=(MODEL,),
        show_fields=None,
        pull_updates=(),
        failures=None,
        redirects=(),
        done_reason="stop",
        reachable=True,
        pull_gate=None,
    ):
        self.model = model or ScriptedModel()
        self.installed = installed
        self.show_fields = show_fields or {}
        self.pull_updates = pull_updates
        self.failures = failures or {}
        self.redirects = redirects
        self.done_reason = done_reason
        self.reachable = reachable
        self.pull_gate = pull_gate
        self.loaded = False
        self.requests = []

    def http_open(self, request):
        if not self.reachable or request.host != LOCAL_HOST:
            raise urllib.error.URLError(f"No route to {request.host}.")
        path = urllib.parse.urlsplit(request.full_url).path
        body = json.loads(request.data) if request.data else None
        self.requests.append(
            {
                "method": request.get_method(),
                "path": path,
                "body": body,
                "content_type": request.get_header("Content-type"),
                "timeout": request.timeout,
            }
        )
        if path in self.redirects:
            return self.respond(request, 302, b"", location=path + MOVED)
        if path in self.failures:
            return self.respond(request, 500, self.failures[path])
        endpoints = {
            ("GET", "/api/tags"): self.tags,
            ("POST", "/api/show"): self.show,
            ("POST", "/api/chat"): self.chat,
            ("POST", "/api/generate"): self.generate,
            ("POST", "/api/pull"): self.pull,
        }
        endpoint = endpoints.get((request.get_method(), path.removesuffix(MOVED)))
        if endpoint is None:
            return self.error(request, 405, "method not allowed")
        return endpoint(request, body)

    def tags(self, request, body):
        models = [{"name": name, "digest": DIGEST} for name in self.installed]
        return self.reply(request, {"models": models})

    def show(self, request, body):
        if not self.knows(body):
            return self.error(request, 404, "model not found")
        details = {"quantization_level": QUANTIZATION}
        return self.reply(request, {"details": details, **self.show_fields})

    def chat(self, request, body):
        if not self.knows(body):
            return self.error(request, 404, "model not found")
        messages = body.get("messages")
        if not isinstance(messages, list) or not all(
            m.get("role") in CHAT_ROLES for m in messages
        ):
            return self.error(request, 400, "invalid message role")
        self.loaded = True
        content = self.model_reply(body, messages)
        if body.get("stream", True) is not False:
            partial = {"message": {"content": content[:1]}, "done": False}
            final = {"message": {"content": content[1:]}, "done": True}
            return self.stream(request, [partial, final])
        message = {"role": "assistant", "content": content}
        return self.reply(
            request, {"message": message, "done_reason": self.done_reason}
        )

    def model_reply(self, body, messages):
        """Without the app's instruction and schema, the model answers in prose."""
        roles = {m.get("role"): m.get("content") for m in messages}
        instruction = roles.get("system")
        if instruction not in (EDIT_PROMPT, REVIEW_PROMPT):
            return PROSE_REPLY
        if not isinstance(body.get("format"), dict):
            return PROSE_REPLY
        return json.dumps(self.model.answer(instruction, json.loads(roles["user"])))

    def generate(self, request, body):
        if not self.knows(body):
            return self.error(request, 404, "model not found")
        self.loaded = body.get("keep_alive") != 0
        return self.reply(request, {"done": True})

    def pull(self, request, body):
        if (body or {}).get("model") != MODEL:
            return self.error(request, 404, "model not found in registry")
        if MODEL not in self.installed:
            self.installed = (*self.installed, MODEL)
        if body.get("stream", True) is False:
            return self.reply(request, {"status": "success"})
        if self.pull_gate is not None:
            lines = b"\n".join(json.dumps(u).encode() for u in self.pull_updates)
            stream = GatedStream(lines, self.pull_gate)
            return self.respond(request, 200, b"", stream=stream)
        return self.stream(request, self.pull_updates)

    def knows(self, body):
        return (body or {}).get("model") in self.installed

    def reply(self, request, value):
        return self.respond(request, 200, json.dumps(value).encode())

    def stream(self, request, values):
        lines = b"\n".join(json.dumps(value).encode() for value in values)
        return self.respond(request, 200, lines)

    def error(self, request, code, message):
        return self.respond(request, code, json.dumps({"error": message}).encode())

    def respond(self, request, code, payload, location=None, stream=None):
        headers = Message()
        headers["Content-Type"] = "application/json"
        if location:
            headers["Location"] = location
        response = urllib.response.addinfourl(
            stream or io.BytesIO(payload), headers, request.full_url, code
        )
        response.msg = "OK" if code < 400 else "Error"
        return response

    def chat_requests(self):
        return [r for r in self.requests if r["path"] == "/api/chat"]


class GatedStream(io.BytesIO):
    """A streamed reply that stalls after its first line until the gate opens."""

    def __init__(self, payload, gate):
        super().__init__(payload)
        self.gate = gate
        self.lines_read = 0

    def readline(self, size=-1):
        if self.lines_read == 1:
            self.gate.wait()
        self.lines_read += 1
        return super().readline(size)

    def __next__(self):
        line = self.readline()
        if not line:
            raise StopIteration
        return line


@contextlib.contextmanager
def served_by(service):
    """Every OllamaClient built inside the block talks to this service."""
    build = urllib.request.build_opener

    def opener_with_service(*handlers):
        return build(*handlers, service)

    with patch("urllib.request.build_opener", opener_with_service):
        yield service
