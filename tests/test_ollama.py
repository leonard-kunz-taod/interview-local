# German fixtures intentionally exercise non-ASCII speech.
import copy
import json
import os
import unittest
from unittest.mock import patch

from support.fake_ollama import IDENTITY, FakeOllama, ScriptedModel, served_by

from interview_local.core import Turn, Word, merge
from interview_local.llm import (
    EDIT_PROMPT,
    MODEL,
    MODEL_SCHEMA,
    REQUEST_TIMEOUT_SECONDS,
    REVIEW_PROMPT,
    REVIEW_SCHEMA,
    OllamaClient,
    enhance,
    review,
)

DOWNLOAD_NOTICE = (
    f"Downloading {MODEL} from the Ollama registry (no HF account needed) …"
)
EDITOR_OPTIONS = {"temperature": 0.1, "num_ctx": 8192, "num_predict": 3072}
ERROR_DETAIL_BYTES = 2000


def silent(_message):
    pass


def ready_from(service):
    with served_by(service):
        return OllamaClient().ready()


def prepared_by(service, status=silent):
    with served_by(service):
        return OllamaClient().prepare(status)


def pulled(*updates):
    messages = []
    prepared_by(FakeOllama(pull_updates=updates), messages.append)
    return messages


def german_source():
    return merge(
        [
            Word(0.12345, 0.4567, "Grüße", 0.9, "whisper"),
            Word(0.4567, 0.9, "aus", 0.9, "whisper"),
        ],
        [Turn(0, 1, "s")],
        [],
    )


def counted_source(count):
    words = [
        Word(i * 0.1, i * 0.1 + 0.05, f"w{i}", 0.9, "whisper") for i in range(count)
    ]
    return merge(words, [Turn(0, count, "s")], [])


def edit_requests(source, model=None):
    service = FakeOllama(model)
    with served_by(service):
        enhance(source, silent)
    return service


def user_data(chat_request):
    return json.loads(chat_request["body"]["messages"][1]["content"])


def word_ids(words):
    return [w["id"] for w in words]


class ReadinessTests(unittest.TestCase):
    def test_ready_returns_the_installed_model_identity(self):
        self.assertEqual(ready_from(FakeOllama()), IDENTITY)

    def test_missing_model_is_reported(self):
        with self.assertRaises(RuntimeError):
            ready_from(FakeOllama(installed=()))

    def test_cloud_backed_model_is_rejected(self):
        service = FakeOllama(show_fields={"remote_host": "https://cloud.example"})
        with self.assertRaisesRegex(RuntimeError, "Cloud-backed"):
            ready_from(service)

    def test_unreachable_service_states_that_no_cloud_fallback_is_used(self):
        with self.assertRaisesRegex(RuntimeError, "No cloud fallback is used"):
            ready_from(FakeOllama(reachable=False))

    def test_remote_model_reference_is_rejected_as_cloud_backed(self):
        service = FakeOllama(show_fields={"remote_model": "ministral-cloud"})
        with self.assertRaisesRegex(RuntimeError, "Cloud-backed"):
            ready_from(service)

    def test_other_installed_models_do_not_count_as_the_editor(self):
        with self.assertRaisesRegex(RuntimeError, "is not installed locally"):
            ready_from(FakeOllama(installed=("llama3:8b",)))


class ServiceErrorTests(unittest.TestCase):
    def readiness_error(self, service):
        with served_by(service), self.assertRaises(RuntimeError) as failure:
            OllamaClient().ready()
        return str(failure.exception)

    def failing_tags(self, detail):
        return self.readiness_error(FakeOllama(failures={"/api/tags": detail}))

    def test_service_error_names_its_status_and_detail(self):
        message = self.failing_tags(b"out of memory")
        self.assertTrue(message.endswith("(500): out of memory"))

    def test_long_error_detail_is_cut_short(self):
        message = self.failing_tags(b"x" * (ERROR_DETAIL_BYTES + 1000))
        self.assertTrue(message.endswith("(500): " + "x" * ERROR_DETAIL_BYTES))

    def test_undecodable_error_detail_is_replaced_not_fatal(self):
        message = self.failing_tags(b"\xffboom")
        self.assertTrue(message.endswith("(500): �boom"))

    def test_unreachable_service_names_the_cause(self):
        message = self.readiness_error(FakeOllama(reachable=False))
        self.assertIn("No route to 127.0.0.1:11434.", message)


class NetworkBoundaryTests(unittest.TestCase):
    def test_redirect_from_the_service_is_refused(self):
        with self.assertRaises(RuntimeError):
            ready_from(FakeOllama(redirects=("/api/tags",)))

    def test_environment_proxies_are_bypassed(self):
        proxy = "http://proxy.invalid:3128"
        environment = {
            "http_proxy": proxy,
            "HTTP_PROXY": proxy,
            "no_proxy": "",
            "NO_PROXY": "",
        }
        with patch.dict(os.environ, environment):
            self.assertEqual(ready_from(FakeOllama()), IDENTITY)


class PreparationTests(unittest.TestCase):
    def test_each_distinct_download_progress_step_is_reported_once(self):
        self.assertEqual(
            pulled(
                {"status": "pulling", "total": 100, "completed": 0},
                {"status": "pulling", "total": 100, "completed": 0},
                {"status": "pulling", "total": 100, "completed": 50},
                {"status": "success"},
            ),
            [DOWNLOAD_NOTICE, "pulling 0%", "pulling 50%", "success"],
        )

    def test_complete_download_shows_100_percent(self):
        update = {"status": "pulling", "total": 100, "completed": 100}
        self.assertEqual(pulled(update), [DOWNLOAD_NOTICE, "pulling 100%"])

    def test_download_without_progress_yet_shows_0_percent(self):
        update = {"status": "pulling", "total": 100}
        self.assertEqual(pulled(update), [DOWNLOAD_NOTICE, "pulling 0%"])

    def test_first_status_is_reported_even_when_empty(self):
        updates = ({"status": ""}, {"status": "success"})
        self.assertEqual(pulled(*updates), [DOWNLOAD_NOTICE, "", "success"])

    def test_download_error_from_the_service_is_raised(self):
        service = FakeOllama(pull_updates=({"error": "disk full"},))
        with self.assertRaisesRegex(RuntimeError, "disk full"):
            prepared_by(service)

    def test_downloaded_model_is_ready_afterwards(self):
        self.assertEqual(prepared_by(FakeOllama(installed=())), IDENTITY)

    def test_failed_download_names_the_cause(self):
        with self.assertRaises(RuntimeError) as failure:
            prepared_by(FakeOllama(reachable=False))
        self.assertIn("No route to 127.0.0.1:11434.", str(failure.exception))


class ChatTests(unittest.TestCase):
    def test_chat_returns_the_parsed_model_json(self):
        service = FakeOllama(ScriptedModel(response={"a": 1}))
        with served_by(service):
            answer = OllamaClient().chat(EDIT_PROMPT, {}, MODEL_SCHEMA)
        self.assertEqual(answer, {"a": 1})

    def test_truncated_model_output_is_rejected(self):
        with (
            served_by(FakeOllama(ScriptedModel(response={}), done_reason="length")),
            self.assertRaises(ValueError),
        ):
            OllamaClient().chat(EDIT_PROMPT, {}, MODEL_SCHEMA)


class EditRequestContractTests(unittest.TestCase):
    """What the editor receives over the Ollama chat API."""

    def test_edit_request_follows_the_ollama_chat_contract(self):
        request = edit_requests(german_source()).chat_requests()[0]
        body = dict(request["body"], messages=None)
        schema = copy.deepcopy(MODEL_SCHEMA)
        schema["properties"]["paragraphs"]["items"]["properties"]["speaker"]["enum"] = [
            "A",
            None,
        ]
        self.assertEqual(
            (body, request["body"]["messages"][0]),
            (
                {
                    "model": MODEL,
                    "stream": False,
                    "messages": None,
                    "format": schema,
                    "options": EDITOR_OPTIONS,
                    "keep_alive": "5m",
                },
                {"role": "system", "content": EDIT_PROMPT},
            ),
        )

    def test_editor_receives_the_words_with_their_metadata_verbatim(self):
        request = edit_requests(german_source()).chat_requests()[0]
        user = request["body"]["messages"][1]
        self.assertEqual(
            (user["role"], "Grüße" in user["content"], json.loads(user["content"])),
            (
                "user",
                True,
                {
                    "speakers": {"A": "Speaker A"},
                    "editable_range": [0, 1],
                    "editable_words": [
                        {
                            "id": 0,
                            "t": "Grüße",
                            "s": "A",
                            "u": False,
                            "o": False,
                            "time": [0.123, 0.457],
                        },
                        {
                            "id": 1,
                            "t": "aus",
                            "s": "A",
                            "u": False,
                            "o": False,
                            "time": [0.457, 0.9],
                        },
                    ],
                    "context_before": [],
                    "context_after": [],
                },
            ),
        )

    def test_editor_sees_twelve_neighbouring_words_on_each_side(self):
        requests = edit_requests(counted_source(130)).chat_requests()
        context = [
            (word_ids(data["context_before"]), word_ids(data["context_after"]))
            for data in map(user_data, requests)
        ]
        self.assertEqual(
            context,
            [([], list(range(120, 130))), (list(range(108, 120)), [])],
        )


class ReviewRequestContractTests(unittest.TestCase):
    def test_review_request_shows_the_source_and_the_edit(self):
        with served_by(FakeOllama()):
            draft = enhance(german_source(), silent)
        service = FakeOllama()
        with served_by(service):
            review(draft, silent)
        request = service.chat_requests()[0]
        body = request["body"]
        self.assertEqual(
            (body["messages"][0], body["format"], user_data(request)),
            (
                {"role": "system", "content": REVIEW_PROMPT},
                REVIEW_SCHEMA,
                {
                    "paragraphs": [
                        {
                            "paragraph_id": 0,
                            "source_text": "Grüße aus",
                            "source_words": user_data(
                                edit_requests(german_source()).chat_requests()[0]
                            )["editable_words"],
                            "edited_text": "Grüße aus",
                            "edited_speaker": "A",
                            "existing_uncertainty": False,
                        }
                    ]
                },
            ),
        )


class RequestEnvelopeContractTests(unittest.TestCase):
    def session(self):
        service = FakeOllama()
        with served_by(service):
            draft = enhance(german_source(), silent)
            review(draft, silent)
            OllamaClient().prepare(silent)
        return service.requests

    def test_every_request_waits_a_bounded_time(self):
        timeouts = {(r["method"], r["path"]): r["timeout"] for r in self.session()}
        self.assertEqual(
            timeouts,
            {
                ("GET", "/api/tags"): 10,
                ("POST", "/api/show"): 30,
                ("POST", "/api/chat"): REQUEST_TIMEOUT_SECONDS,
                ("POST", "/api/generate"): 30,
                ("POST", "/api/pull"): REQUEST_TIMEOUT_SECONDS,
            },
        )

    def test_every_request_body_is_declared_json(self):
        content_types = {r["content_type"] for r in self.session() if r["body"]}
        self.assertEqual(content_types, {"application/json"})

    def test_unload_request_drops_the_model_from_memory(self):
        unloads = [r["body"] for r in self.session() if r["path"] == "/api/generate"]
        self.assertEqual(unloads, [{"model": MODEL, "keep_alive": 0}] * 2)


if __name__ == "__main__":
    unittest.main()
