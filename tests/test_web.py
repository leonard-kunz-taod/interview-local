from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from datetime import datetime
from http.client import HTTPResponse
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from interview_local.artifacts import save_readable
from interview_local.web.server import (EditorRequestHandler, EditorState, HttpError, RangeNotSatisfiable,
                                        create_run_dir, parse_byte_range, resolve_run_dir, serve, static_file)
from run_fixtures import TWO_PARAGRAPHS, ScriptedClient, readable_transcript, speech_models

PORT = 8765
HOST = f"127.0.0.1:{PORT}"
RUN_ID = "2026-01-01_10-00-00-000000"
AUDIO_BYTES = bytes(range(256)) * 4
JOB_TIMEOUT_SECONDS = 10


class InMemoryConnection:
    """A socket stand-in carrying one raw HTTP exchange."""

    def __init__(self, raw_request: bytes):
        self.request = io.BytesIO(raw_request)
        self.response = bytearray()

    def makefile(self, mode, *args, **kwargs):
        return self.request if "r" in mode else io.BytesIO()

    def sendall(self, data):
        self.response += data


class ResponseSocket:
    def __init__(self, data: bytes):
        self.data = data

    def makefile(self, mode, *args, **kwargs):
        return io.BytesIO(self.data)


class FakeServer:
    def __init__(self, runs_root: Path):
        self.state = EditorState(runs_root)
        self.server_address = ("127.0.0.1", PORT)


def exchange(server, method, path, body=b"", headers=None):
    """Send one request through the real handler and return (status, headers, body)."""
    headers = {"Host": HOST, **(headers or {})}
    if body or method != "GET":
        headers.setdefault("Content-Length", str(len(body)))
    connection = InMemoryConnection(request_head(method, path, headers) + body)
    with patch.object(EditorRequestHandler, "log_message"):
        EditorRequestHandler(connection, ("127.0.0.1", 50000), server)
    response = HTTPResponse(ResponseSocket(bytes(connection.response)), method=method)
    response.begin()
    return response.status, response, response.read()


def request_head(method, path, headers):
    lines = [f"{method} {path} HTTP/1.1"] + [f"{name}: {value}" for name, value in headers.items()]
    return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")


def raw_exchange(server, method, path, body=b"", headers=None, trailing=b""):
    """Send one request, followed by trailing bytes on the same connection; return the raw response bytes."""
    headers = {"Host": HOST, "Content-Length": str(len(body)), **(headers or {})}
    connection = InMemoryConnection(request_head(method, path, headers) + body + trailing)
    with patch.object(EditorRequestHandler, "log_message"):
        EditorRequestHandler(connection, ("127.0.0.1", 50000), server)
    return bytes(connection.response)


def raw_body(raw_response):
    return raw_response.partition(b"\r\n\r\n")[2]


def send_json(server, method, path, value):
    return exchange(server, method, path, json.dumps(value).encode(), {"Content-Type": "application/json"})


def finished_job(server, job_id):
    """Poll the job endpoint until the job leaves the running state; return its last status."""
    deadline = time.monotonic() + JOB_TIMEOUT_SECONDS
    while True:
        job = json.loads(exchange(server, "GET", f"/api/jobs/{job_id}")[2])
        if job["state"] != "running" or time.monotonic() > deadline:
            return job
        time.sleep(.02)


def stage_file_contents(run_dir):
    return {p.name: p.read_bytes() for p in run_dir.iterdir() if p.name.startswith(("01_", "02_", "03_"))}


class InterruptedServer(FakeServer):
    """A server stand-in that needs no port and is stopped by Ctrl+C at once."""

    def serve_forever(self):
        raise KeyboardInterrupt

    def server_close(self):
        pass


def interrupted_serve(runs_root):
    """Run serve until an immediate Ctrl+C; return (printed output, opened browser URLs)."""
    opened = []
    output = io.StringIO()
    with (patch("interview_local.web.server.webbrowser.open", opened.append),
          patch("interview_local.web.server.EditorServer", InterruptedServer),
          redirect_stdout(output)):
        serve(runs_root)
    return output.getvalue(), opened


class ServeTests(unittest.TestCase):
    def test_browser_opens_the_printed_loopback_url(self):
        with tempfile.TemporaryDirectory() as directory:
            printed, opened = interrupted_serve(Path(directory))
        url = f"http://{HOST}/"
        self.assertEqual((url in printed, opened), (True, [url]))

    def test_missing_runs_directory_is_created_with_its_parents(self):
        with tempfile.TemporaryDirectory() as directory:
            runs_root = Path(directory) / "interviews" / "runs"
            interrupted_serve(runs_root)
            self.assertTrue(runs_root.is_dir())


class FixedClock:
    """Stands in for datetime: now() is always the same upload time."""
    UPLOAD_TIME = datetime(2026, 1, 2, 9, 30, 15, 250)

    @classmethod
    def now(cls):
        return cls.UPLOAD_TIME


class RunDirectoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.runs_root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def test_run_directory_is_named_after_its_creation_time(self):
        with patch("interview_local.web.server.datetime", FixedClock):
            run_dir = create_run_dir(self.runs_root)
        self.assertEqual(run_dir, self.runs_root / "2026-01-02_09-30-15-000250")

    def test_run_directory_is_created_when_the_runs_root_is_missing(self):
        run_dir = create_run_dir(self.runs_root / "deleted")
        self.assertTrue(run_dir.is_dir())

    def test_run_id_with_a_backslash_is_rejected(self):
        (self.runs_root / "other\\run").mkdir()
        with self.assertRaises(HttpError):
            resolve_run_dir(self.runs_root, "other\\run")

    def test_run_id_that_is_a_path_to_a_run_is_rejected(self):
        (self.runs_root / RUN_ID).mkdir()
        with self.assertRaises(HttpError):
            resolve_run_dir(self.runs_root, f"other/../{RUN_ID}")


class ByteRangeTests(unittest.TestCase):
    def test_closed_range(self):
        self.assertEqual(parse_byte_range("bytes=10-19", 100), (10, 19))

    def test_open_range_ends_at_the_last_byte(self):
        self.assertEqual(parse_byte_range("bytes=10-", 100), (10, 99))

    def test_range_end_is_clamped_to_the_file(self):
        self.assertEqual(parse_byte_range("bytes=90-200", 100), (90, 99))

    def test_suffix_range_returns_the_last_bytes(self):
        self.assertEqual(parse_byte_range("bytes=-10", 100), (90, 99))

    def test_single_byte_range(self):
        self.assertEqual(parse_byte_range("bytes=5-5", 100), (5, 5))

    def test_suffix_longer_than_the_file_returns_the_whole_file(self):
        self.assertEqual(parse_byte_range("bytes=-200", 100), (0, 99))

    def test_suffix_range_of_an_empty_file_is_not_satisfiable(self):
        with self.assertRaises(RangeNotSatisfiable):
            parse_byte_range("bytes=-10", 0)

    def test_range_beyond_the_file_is_not_satisfiable(self):
        for header in ("bytes=100-", "bytes=-0", "bytes=20-10"):
            with self.subTest(header=header), self.assertRaises(RangeNotSatisfiable):
                parse_byte_range(header, 100)

    def test_malformed_or_multiple_ranges_are_rejected(self):
        for header in ("bytes=-", "items=0-1", "bytes=0-1,5-6", "bytes=a-b"):
            with self.subTest(header=header), self.assertRaises(ValueError):
                parse_byte_range(header, 100)


class StaticFileTests(unittest.TestCase):
    def test_frontend_files_are_found_as_package_data(self):
        for name in ("index.html", "app.js", "style.css"):
            with self.subTest(name=name):
                self.assertTrue(static_file(name))

    def test_index_is_served_at_the_root(self):
        with tempfile.TemporaryDirectory() as directory:
            status, response, body = exchange(FakeServer(Path(directory)), "GET", "/")
        self.assertEqual((status, b'accept=".wav,.mp3,.m4a"' in body), (200, True))

    def test_frontend_script_is_served_with_its_content_type(self):
        with tempfile.TemporaryDirectory() as directory:
            status, response, _ = exchange(FakeServer(Path(directory)), "GET", "/static/app.js")
        self.assertEqual((status, response.getheader("Content-Type")), (200, "text/javascript; charset=utf-8"))

    def test_unknown_static_file_is_not_found(self):
        with tempfile.TemporaryDirectory() as directory:
            status, _, _ = exchange(FakeServer(Path(directory)), "GET", "/static/secret.py")
        self.assertEqual(status, 404)


class RequestGuardTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.server = FakeServer(Path(self.directory.name))

    def tearDown(self):
        self.directory.cleanup()

    def test_foreign_host_is_rejected(self):
        status, _, _ = exchange(self.server, "GET", "/api/runs", headers={"Host": f"evil.example:{PORT}"})
        self.assertEqual(status, 403)

    def test_localhost_host_is_accepted(self):
        status, _, body = exchange(self.server, "GET", "/api/runs", headers={"Host": f"localhost:{PORT}"})
        self.assertEqual((status, json.loads(body)), (200, {"runs": []}))

    def test_write_without_json_or_audio_header_is_rejected(self):
        status, _, _ = exchange(self.server, "PUT", f"/api/runs/{RUN_ID}/speakers", b'{"A": "x"}',
                                {"Content-Type": "text/plain"})
        self.assertEqual(status, 415)

    def test_write_without_any_content_type_is_rejected(self):
        status, _, _ = exchange(self.server, "PUT", f"/api/runs/{RUN_ID}/speakers", b'{"A": "x"}')
        self.assertEqual(status, 415)

    def test_json_response_declares_its_type_length_and_no_caching(self):
        _, response, body = exchange(self.server, "GET", "/api/runs")
        headers = (response.getheader("Content-Type"), response.getheader("Content-Length"),
                   response.getheader("Cache-Control"))
        self.assertEqual(headers, ("application/json; charset=utf-8", str(len(body)), "no-store"))

    def test_unknown_route_is_not_found(self):
        for method, path in (("GET", "/api/unknown"), ("POST", "/"), ("PUT", f"/api/runs/{RUN_ID}/document"),
                             ("GET", f"/api/runs/{RUN_ID}/export.pdf"), ("GET", "/api/runs/x/y/z/document")):
            with self.subTest(method=method, path=path):
                status, _, _ = exchange(self.server, method, path, headers={"Content-Type": "application/json"})
                self.assertEqual(status, 404)

    def test_unknown_run_is_not_found(self):
        status, _, _ = exchange(self.server, "GET", "/api/runs/2020-01-01_00-00-00-000000/document")
        self.assertEqual(status, 404)

    def test_path_traversal_run_id_is_rejected(self):
        for run_id in ("..", "%2E%2E", "..%2Fetc"):
            with self.subTest(run_id=run_id):
                status, _, _ = exchange(self.server, "GET", f"/api/runs/{run_id}/document")
                self.assertEqual(status, 404)


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.runs_root = Path(self.directory.name)
        self.server = FakeServer(self.runs_root)

    def tearDown(self):
        self.directory.cleanup()

    def upload(self, filename, body=b"RIFF", query="name_a=Anna&name_b=Bernd"):
        return exchange(self.server, "POST", f"/api/runs?{query}", body,
                        {"X-Audio-Filename": filename, "Content-Type": "application/octet-stream"})

    def test_unsupported_format_is_rejected_with_an_understandable_message(self):
        for filename in ("interview.flac", "interview.ogg", "interview", "notes.wav.txt"):
            with self.subTest(filename=filename):
                status, _, body = self.upload(filename)
                self.assertEqual((status, json.loads(body)["error"]), (400, "Please select a WAV, MP3 or M4A file."))

    def test_unsupported_format_creates_no_run(self):
        self.upload("interview.flac")
        self.assertEqual(list(self.runs_root.iterdir()), [])

    def test_missing_speaker_name_is_rejected(self):
        status, _, _ = self.upload("interview.wav", query="name_a=Anna&name_b=")
        self.assertEqual(status, 400)

    def test_supported_upload_is_stored_as_the_run_source(self):
        with patch("interview_local.web.jobs.TranscriptionJob.start"):
            status, _, body = self.upload("Gespr%C3%A4ch.M4A", body=AUDIO_BYTES)
        stored = (self.runs_root / json.loads(body)["run_id"] / "source.m4a").read_bytes()
        self.assertEqual((status, stored), (202, AUDIO_BYTES))

    def test_supported_upload_starts_a_running_job(self):
        with patch("interview_local.web.jobs.TranscriptionJob.start"):
            _, _, body = self.upload("interview.wav")
        job_status, _, job = exchange(self.server, "GET", f"/api/jobs/{json.loads(body)['job_id']}")
        self.assertEqual((job_status, json.loads(job)["state"]), (200, "running"))

    def test_unknown_job_is_not_found(self):
        status, _, _ = exchange(self.server, "GET", "/api/jobs/unknown")
        self.assertEqual(status, 404)

    def test_second_upload_while_a_job_runs_is_rejected(self):
        with patch("interview_local.web.jobs.TranscriptionJob.start"):
            self.upload("first.wav")
            status, _, _ = self.upload("second.wav")
        self.assertEqual(status, 409)

    def test_upload_while_another_upload_is_received_is_rejected(self):
        with self.server.state.upload_lock:
            status, _, _ = self.upload("interview.wav")
        self.assertEqual(status, 409)

    def test_empty_upload_is_rejected_without_a_run(self):
        status, _, _ = self.upload("interview.wav", body=b"")
        self.assertEqual((status, list(self.runs_root.iterdir())), (400, []))

    def test_one_byte_upload_is_accepted(self):
        with patch("interview_local.web.jobs.TranscriptionJob.start"):
            status, _, _ = self.upload("interview.wav", body=b"R")
        self.assertEqual(status, 202)

    def test_upload_without_a_speaker_name_is_rejected(self):
        for case, query in (("no names", ""), ("only the second name", "name_b=Bernd")):
            with self.subTest(case):
                status, _, _ = self.upload("interview.wav", query=query)
                self.assertEqual(status, 400)

    def test_upload_without_a_filename_is_rejected(self):
        status, _, _ = exchange(self.server, "POST", "/api/runs?name_a=Anna&name_b=Bernd", AUDIO_BYTES,
                                {"Content-Type": "application/json"})
        self.assertEqual(status, 400)

    def test_upload_that_ends_early_is_rejected_without_a_run(self):
        raw = raw_exchange(self.server, "POST", "/api/runs?name_a=Anna&name_b=Bernd", AUDIO_BYTES[:10],
                           {"X-Audio-Filename": "interview.wav", "Content-Length": str(len(AUDIO_BYTES))})
        self.assertEqual((raw.split()[1], list(self.runs_root.iterdir())), (b"400", []))

    def test_upload_stores_exactly_the_announced_length(self):
        with patch("interview_local.web.jobs.TranscriptionJob.start"):
            raw = raw_exchange(self.server, "POST", "/api/runs?name_a=Anna&name_b=Bernd", AUDIO_BYTES,
                               {"X-Audio-Filename": "interview.wav"}, trailing=b"next request")
        stored = (self.runs_root / json.loads(raw_body(raw))["run_id"] / "source.wav").read_bytes()
        self.assertEqual(stored, AUDIO_BYTES)

    def test_job_endpoint_reports_the_job_status(self):
        with patch("interview_local.web.jobs.TranscriptionJob.start"):
            _, _, body = self.upload("interview.wav")
        job_id = json.loads(body)["job_id"]
        _, _, job = exchange(self.server, "GET", f"/api/jobs/{job_id}")
        self.assertEqual(json.loads(job), asdict(self.server.state.jobs[job_id].status()))

    def transcribed_upload(self, failing_stage=None):
        """Upload a recording, let its job run against the speech-model and LLM stand-ins; return
        (upload response, final job status)."""
        with (patch("interview_local.pipeline.worker", speech_models(failing_stage)),
              patch("interview_local.web.jobs.OllamaClient", lambda: ScriptedClient([(0, 1, "A", "Guten Tag.")])),
              patch("interview_local.models.local_diarization_model", lambda: None),
              patch.dict(os.environ, {"HF_TOKEN": "hf_test"})):
            _, _, body = self.upload("interview.wav", body=AUDIO_BYTES)
            upload = json.loads(body)
            return upload, finished_job(self.server, upload["job_id"])

    def test_uploaded_recording_is_transcribed_with_the_entered_speaker_names(self):
        upload, job = self.transcribed_upload()
        _, _, document = exchange(self.server, "GET", f"/api/runs/{upload['run_id']}/document")
        self.assertEqual((job["state"], json.loads(document)["speakers"]), ("done", {"A": "Anna", "B": "Bernd"}))

    def test_failed_job_reports_its_error(self):
        _, job = self.transcribed_upload(failing_stage="asr")
        self.assertEqual((job["state"], "model unavailable" in job["error"]), ("failed", True))


class EditorApiTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.runs_root = Path(self.directory.name)
        self.run_dir = self.runs_root / RUN_ID
        save_readable(readable_transcript(TWO_PARAGRAPHS), self.run_dir)
        (self.run_dir / "source.wav").write_bytes(AUDIO_BYTES)
        self.server = FakeServer(self.runs_root)

    def tearDown(self):
        self.directory.cleanup()

    def document(self):
        status, _, body = exchange(self.server, "GET", f"/api/runs/{RUN_ID}/document")
        self.assertEqual(status, 200)
        return json.loads(body)

    def test_runs_with_readable_transcript_are_listed(self):
        (self.runs_root / "unfinished").mkdir()
        _, _, body = exchange(self.server, "GET", "/api/runs")
        self.assertEqual(json.loads(body), {"runs": [{"run_id": RUN_ID}]})

    def test_document_lists_paragraphs_with_names(self):
        document = self.document()
        self.assertEqual([p["speaker_name"] for p in document["paragraphs"]], ["Anna", "Bernd"])

    def test_run_without_stored_audio_cannot_be_opened(self):
        (self.run_dir / "source.wav").unlink()
        status, _, body = exchange(self.server, "GET", f"/api/runs/{RUN_ID}/document")
        self.assertEqual((status, "audio" in json.loads(body)["error"]), (409, True))

    def test_text_edit_is_saved_to_corrections_and_reloaded(self):
        status, _, _ = send_json(self.server, "PUT", f"/api/runs/{RUN_ID}/paragraphs/4/text",
                                 {"text": "Danke, sehr gut."})
        reloaded = self.document()["paragraphs"][1]["text"]
        saved = (self.run_dir / "corrections.json").is_file()
        self.assertEqual((status, reloaded, saved), (200, "Danke, sehr gut.", True))

    def test_empty_text_edit_is_rejected(self):
        status, _, _ = send_json(self.server, "PUT", f"/api/runs/{RUN_ID}/paragraphs/4/text", {"text": " "})
        self.assertEqual(status, 400)

    def test_reassignment_splits_the_paragraph(self):
        status, _, body = send_json(self.server, "POST", f"/api/runs/{RUN_ID}/reassign",
                                    {"first_word": 5, "last_word": 6, "speaker": "A"})
        names = [p["speaker_name"] for p in json.loads(body)["paragraphs"]]
        self.assertEqual((status, names), (200, ["Anna", "Bernd", "Anna", "Bernd"]))

    def test_speaker_names_can_be_swapped(self):
        send_json(self.server, "PUT", f"/api/runs/{RUN_ID}/speakers", {"A": "Bernd", "B": "Anna"})
        self.assertEqual([p["speaker_name"] for p in self.document()["paragraphs"]], ["Bernd", "Anna"])

    def test_speaker_name_that_is_no_text_is_rejected(self):
        status, _, body = send_json(self.server, "PUT", f"/api/runs/{RUN_ID}/speakers", {"A": "Anna", "B": 7})
        self.assertEqual((status, json.loads(body)["error"]), (400, "Speaker names must be text."))

    def test_stage_files_stay_unchanged_after_corrections(self):
        before = stage_file_contents(self.run_dir)
        send_json(self.server, "PUT", f"/api/runs/{RUN_ID}/paragraphs/4/text", {"text": "Danke."})
        self.assertEqual(stage_file_contents(self.run_dir), before)

    def test_export_is_a_corrected_download(self):
        send_json(self.server, "PUT", f"/api/runs/{RUN_ID}/paragraphs/4/text", {"text": "Danke, sehr gut."})
        for export, first_line in (("export.md", "# Corrected transcript"), ("export.txt", "Corrected transcript")):
            with self.subTest(export=export):
                status, response, body = exchange(self.server, "GET", f"/api/runs/{RUN_ID}/{export}")
                text = body.decode("utf-8")
                download = (status, response.getheader("Content-Disposition").startswith("attachment"),
                            text.splitlines()[0], "Danke, sehr gut." in text)
                self.assertEqual(download, (200, True, first_line, True))

    def test_audio_range_request_returns_partial_content(self):
        status, response, body = exchange(self.server, "GET", f"/api/runs/{RUN_ID}/audio",
                                          headers={"Range": "bytes=10-19"})
        partial = (status, body, response.getheader("Content-Range"), response.getheader("Content-Type"))
        self.assertEqual(partial, (206, AUDIO_BYTES[10:20], f"bytes 10-19/{len(AUDIO_BYTES)}", "audio/wav"))

    def test_audio_range_beyond_the_file_is_not_satisfiable(self):
        status, _, _ = exchange(self.server, "GET", f"/api/runs/{RUN_ID}/audio",
                                headers={"Range": f"bytes={len(AUDIO_BYTES)}-"})
        self.assertEqual(status, 416)

    def test_run_with_two_stored_audio_copies_cannot_be_opened(self):
        (self.run_dir / "source.mp3").write_bytes(AUDIO_BYTES)
        status, _, _ = exchange(self.server, "GET", f"/api/runs/{RUN_ID}/document")
        self.assertEqual(status, 409)

    def test_broken_readable_transcript_is_an_internal_error(self):
        (self.run_dir / "03_readable.json").write_text("{}", encoding="utf-8")
        with redirect_stderr(io.StringIO()):
            status, _, body = exchange(self.server, "GET", f"/api/runs/{RUN_ID}/document")
        self.assertEqual((status, json.loads(body)["error"]), (500, "KeyError: 'paragraphs'"))

    def test_runs_are_listed_newest_first(self):
        later = self.runs_root / "2026-02-01_10-00-00-000000"
        save_readable(readable_transcript(TWO_PARAGRAPHS), later)
        _, _, body = exchange(self.server, "GET", "/api/runs")
        self.assertEqual(json.loads(body), {"runs": [{"run_id": later.name}, {"run_id": RUN_ID}]})

    def test_reassignment_without_integer_word_ids_is_rejected(self):
        status, _, _ = send_json(self.server, "POST", f"/api/runs/{RUN_ID}/reassign",
                                 {"first_word": "5", "last_word": 6, "speaker": "A"})
        self.assertEqual(status, 400)

    def test_audio_without_range_returns_the_whole_file(self):
        status, _, body = exchange(self.server, "GET", f"/api/runs/{RUN_ID}/audio")
        self.assertEqual((status, body), (200, AUDIO_BYTES))

    def test_audio_response_offers_byte_ranges_and_declares_its_length(self):
        _, response, _ = exchange(self.server, "GET", f"/api/runs/{RUN_ID}/audio")
        self.assertEqual((response.getheader("Accept-Ranges"), response.getheader("Content-Length")),
                         ("bytes", str(len(AUDIO_BYTES))))

    def test_audio_range_response_carries_only_the_requested_bytes(self):
        raw = raw_exchange(self.server, "GET", f"/api/runs/{RUN_ID}/audio", headers={"Range": "bytes=10-19"})
        self.assertEqual(raw_body(raw), AUDIO_BYTES[10:20])

    def test_document_names_its_run(self):
        self.assertEqual(self.document()["run_id"], RUN_ID)

    def test_json_body_is_read_only_up_to_its_announced_length(self):
        raw = raw_exchange(self.server, "PUT", f"/api/runs/{RUN_ID}/speakers", b'{"A": "Bernd", "B": "Anna"}',
                           {"Content-Type": "application/json"}, trailing=b"next request")
        self.assertEqual(raw.split()[1], b"200")

    def test_export_declares_its_type_and_length(self):
        _, response, body = exchange(self.server, "GET", f"/api/runs/{RUN_ID}/export.txt")
        self.assertEqual((response.getheader("Content-Type"), response.getheader("Content-Length")),
                         ("text/plain; charset=utf-8", str(len(body))))


if __name__ == "__main__":
    unittest.main()
