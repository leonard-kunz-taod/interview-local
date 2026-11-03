from contextlib import redirect_stdout
from http.client import HTTPResponse
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from interview_local.artifacts import save_readable
from interview_local.web.server import (EditorRequestHandler, EditorState, RangeNotSatisfiable, parse_byte_range,
                                        serve, static_file)
from run_fixtures import TWO_PARAGRAPHS, readable_transcript

PORT = 8765
HOST = f"127.0.0.1:{PORT}"
RUN_ID = "2026-01-01_10-00-00-000000"
AUDIO_BYTES = bytes(range(256)) * 4


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
    head = f"{method} {path} HTTP/1.1\r\n" + "".join(f"{k}: {v}\r\n" for k, v in headers.items()) + "\r\n"
    connection = InMemoryConnection(head.encode("latin-1") + body)
    with patch.object(EditorRequestHandler, "log_message"):
        EditorRequestHandler(connection, ("127.0.0.1", 50000), server)
    response = HTTPResponse(ResponseSocket(bytes(connection.response)), method=method)
    response.begin()
    return response.status, response, response.read()


def send_json(server, method, path, value):
    return exchange(server, method, path, json.dumps(value).encode(), {"Content-Type": "application/json"})


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

    def test_missing_runs_directory_is_created(self):
        with tempfile.TemporaryDirectory() as directory:
            runs_root = Path(directory) / "runs"
            interrupted_serve(runs_root)
            self.assertTrue(runs_root.is_dir())


class ByteRangeTests(unittest.TestCase):
    def test_closed_range(self):
        self.assertEqual(parse_byte_range("bytes=10-19", 100), (10, 19))

    def test_open_range_ends_at_the_last_byte(self):
        self.assertEqual(parse_byte_range("bytes=10-", 100), (10, 99))

    def test_range_end_is_clamped_to_the_file(self):
        self.assertEqual(parse_byte_range("bytes=90-200", 100), (90, 99))

    def test_suffix_range_returns_the_last_bytes(self):
        self.assertEqual(parse_byte_range("bytes=-10", 100), (90, 99))

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

    def test_stage_files_stay_unchanged_after_corrections(self):
        stage_files = sorted(p for p in self.run_dir.iterdir() if p.name.startswith(("01_", "02_", "03_")))
        before = [p.read_bytes() for p in stage_files]
        send_json(self.server, "PUT", f"/api/runs/{RUN_ID}/paragraphs/4/text", {"text": "Danke."})
        self.assertEqual([p.read_bytes() for p in stage_files], before)

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

    def test_audio_without_range_returns_the_whole_file(self):
        status, _, body = exchange(self.server, "GET", f"/api/runs/{RUN_ID}/audio")
        self.assertEqual((status, body), (200, AUDIO_BYTES))


if __name__ == "__main__":
    unittest.main()
