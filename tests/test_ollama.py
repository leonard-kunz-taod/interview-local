import io
import json
import unittest
import urllib.error
import urllib.parse
from unittest.mock import patch

from interview_local.llm import MODEL, OllamaClient

DOWNLOAD_NOTICE = (
    f"Downloading {MODEL} from the Ollama registry (no HF account needed) …"
)


def body(value):
    return json.dumps(value).encode()


def stream(*updates):
    return b"\n".join(body(update) for update in updates)


INSTALLED = {
    "/api/tags": body({"models": [{"name": MODEL, "digest": "abc"}]}),
    "/api/show": body({"details": {"quantization_level": "Q4_K_M"}}),
}


class FakeOllamaService:
    """Replays canned response bodies per endpoint of the local HTTP service."""

    def __init__(self, bodies):
        self.bodies = bodies

    def open(self, request, timeout):
        return io.BytesIO(self.bodies[urllib.parse.urlsplit(request.full_url).path])


class UnreachableService:
    def open(self, request, timeout):
        raise urllib.error.URLError("connection refused")


def client_for(service):
    with patch("urllib.request.build_opener", return_value=service):
        return OllamaClient()


class ReadinessTests(unittest.TestCase):
    def test_ready_returns_the_installed_model_identity(self):
        identity = client_for(FakeOllamaService(INSTALLED)).ready()
        self.assertEqual(
            identity, {"model": MODEL, "digest": "abc", "quantization": "Q4_K_M"}
        )

    def test_missing_model_is_reported(self):
        tags = body({"models": []})
        service = FakeOllamaService(dict(INSTALLED, **{"/api/tags": tags}))
        with self.assertRaises(RuntimeError):
            client_for(service).ready()

    def test_cloud_backed_model_is_rejected(self):
        show = body({"remote_host": "https://cloud.example"})
        service = FakeOllamaService(dict(INSTALLED, **{"/api/show": show}))
        with self.assertRaisesRegex(RuntimeError, "Cloud-backed"):
            client_for(service).ready()

    def test_unreachable_service_states_that_no_cloud_fallback_is_used(self):
        with self.assertRaisesRegex(RuntimeError, "No cloud fallback is used"):
            client_for(UnreachableService()).ready()

    def test_remote_model_reference_is_rejected_as_cloud_backed(self):
        show = body({"remote_model": "ministral-cloud"})
        service = FakeOllamaService(dict(INSTALLED, **{"/api/show": show}))
        with self.assertRaisesRegex(RuntimeError, "Cloud-backed"):
            client_for(service).ready()

    def test_other_installed_models_do_not_count_as_the_editor(self):
        tags = body({"models": [{"name": "llama3:8b", "digest": "zzz"}]})
        service = FakeOllamaService(dict(INSTALLED, **{"/api/tags": tags}))
        with self.assertRaisesRegex(RuntimeError, "is not installed locally"):
            client_for(service).ready()


class PreparationTests(unittest.TestCase):
    def test_each_distinct_download_progress_step_is_reported_once(self):
        pull = stream(
            {"status": "pulling", "total": 100, "completed": 0},
            {"status": "pulling", "total": 100, "completed": 0},
            {"status": "pulling", "total": 100, "completed": 50},
            {"status": "success"},
        )
        messages = []
        service = FakeOllamaService(dict(INSTALLED, **{"/api/pull": pull}))
        client_for(service).prepare(messages.append)
        self.assertEqual(
            messages, [DOWNLOAD_NOTICE, "pulling 0%", "pulling 50%", "success"]
        )

    def test_download_error_from_the_service_is_raised(self):
        pull = stream({"error": "disk full"})
        service = FakeOllamaService(dict(INSTALLED, **{"/api/pull": pull}))
        with self.assertRaisesRegex(RuntimeError, "disk full"):
            client_for(service).prepare(lambda _: None)


class ChatTests(unittest.TestCase):
    def test_chat_returns_the_parsed_model_json(self):
        reply = body({"done_reason": "stop", "message": {"content": '{"a": 1}'}})
        client = client_for(FakeOllamaService({"/api/chat": reply}))
        self.assertEqual(client.chat("instruction", {}, {}), {"a": 1})

    def test_truncated_model_output_is_rejected(self):
        reply = body({"done_reason": "length", "message": {"content": "{}"}})
        client = client_for(FakeOllamaService({"/api/chat": reply}))
        with self.assertRaises(ValueError):
            client.chat("instruction", {}, {})


if __name__ == "__main__":
    unittest.main()
