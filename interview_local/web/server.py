"""Loopback-only HTTP server for uploading recordings and correcting transcripts."""
from __future__ import annotations
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
import json
from pathlib import Path
import re
import shutil
import threading
import traceback
import urllib.parse
import webbrowser

from ..artifacts import corrected_text
from ..corrections import (edit_paragraph_text, load_corrected_paragraphs, load_speaker_names,
                           readable_source_words, reassign_words, save_corrections)
from ..editor_document import build_editor_document
from .jobs import TranscriptionJob

LOOPBACK_HOST = "127.0.0.1"
UPLOAD_CHUNK_BYTES = 1024 * 1024
AUDIO_CONTENT_TYPES = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4"}
UNSUPPORTED_FORMAT_MESSAGE = "Please select a WAV, MP3 or M4A file."
STORED_AUDIO_STEM = "source"
READABLE_FILENAME = "03_readable.json"
AUDIO_FILENAME_HEADER = "X-Audio-Filename"
STATIC_CONTENT_TYPES = {"index.html": "text/html; charset=utf-8", "app.js": "text/javascript; charset=utf-8",
                        "style.css": "text/css; charset=utf-8"}
EXPORT_FORMATS = {"export.md": ("md", True, "text/markdown; charset=utf-8"),
                  "export.txt": ("txt", False, "text/plain; charset=utf-8")}


class HttpError(Exception):
    """A request failure with the HTTP status to report."""

    def __init__(self, status: HTTPStatus, message: str):
        super().__init__(message)
        self.status = status


class RangeNotSatisfiable(ValueError):
    """The requested byte range lies outside the file."""


def serve(runs_root: Path, open_browser: bool) -> None:
    """Serve the web UI on an ephemeral loopback port until Ctrl+C.

    Args:
        runs_root: Directory that holds one subdirectory per run.
        open_browser: Whether to open the UI in the default browser.
    """
    runs_root.mkdir(parents=True, exist_ok=True)
    server = EditorServer(runs_root)
    url = f"http://{LOOPBACK_HOST}:{server.server_address[1]}/"
    print(f"Interview Local web UI: {url}\nRuns are saved in {runs_root}. Press Ctrl+C to stop.", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nWeb UI stopped.")
    finally:
        server.server_close()


class EditorState:
    """Shared server state: the runs directory, jobs and write locks."""

    def __init__(self, runs_root: Path):
        self.runs_root = runs_root.resolve()
        self.jobs: dict[str, TranscriptionJob] = {}
        self.upload_lock = threading.Lock()
        self.corrections_lock = threading.Lock()

    def job_running(self) -> bool:
        """Whether any transcription job is still running."""
        return any(job.running for job in self.jobs.values())


class EditorServer(ThreadingHTTPServer):
    """A threading HTTP server bound to an ephemeral loopback port."""

    daemon_threads = True

    def __init__(self, runs_root: Path):
        super().__init__((LOOPBACK_HOST, 0), EditorRequestHandler)
        self.state = EditorState(runs_root)


def parse_byte_range(header: str, size: int) -> tuple[int, int]:
    """Parse a single-range ``Range`` header.

    Args:
        header: The header value, e.g. ``bytes=0-1023``, ``bytes=500-`` or ``bytes=-500``.
        size: The file size in bytes.

    Returns:
        The inclusive first and last byte positions.

    Raises:
        RangeNotSatisfiable: If the range lies outside the file.
        ValueError: If the header is malformed or requests several ranges.
    """
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
    if not match or match.groups() == ("", ""):
        raise ValueError(f"Unsupported Range header: {header}")
    first, last = match.groups()
    if not first:
        suffix_length = int(last)
        if suffix_length == 0 or size == 0:
            raise RangeNotSatisfiable(f"Range {header} is not satisfiable.")
        return max(0, size - suffix_length), size - 1
    start = int(first)
    end = min(int(last), size - 1) if last else size - 1
    if start >= size or end < start:
        raise RangeNotSatisfiable(f"Range {header} is not satisfiable.")
    return start, end


class EditorRequestHandler(BaseHTTPRequestHandler):
    """Routes API, audio and static requests; rejects foreign hosts and simple cross-site writes."""

    server_version = "InterviewLocal"

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def do_PUT(self) -> None:
        self._dispatch("PUT")

    @property
    def state(self) -> EditorState:
        return self.server.state

    def _dispatch(self, method: str) -> None:
        try:
            self._require_loopback_host()
            if method != "GET":
                self._require_preflighted_write()
            self._route(method, self._path_segments())
        except HttpError as exc:
            self._send_json(exc.status, {"error": str(exc)})
        except RangeNotSatisfiable as exc:
            self._send_json(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, {"error": str(exc)})
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:
            traceback.print_exc()
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(exc).__name__}: {exc}"})

    def _require_loopback_host(self) -> None:
        port = self.server.server_address[1]
        if self.headers.get("Host") not in {f"{LOOPBACK_HOST}:{port}", f"localhost:{port}"}:
            raise HttpError(HTTPStatus.FORBIDDEN, "Requests must address this machine's loopback interface.")

    def _require_preflighted_write(self) -> None:
        content_type = self.headers.get("Content-Type", "")
        if not content_type.startswith("application/json") and AUDIO_FILENAME_HEADER not in self.headers:
            raise HttpError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                            f"Write requests need Content-Type: application/json or {AUDIO_FILENAME_HEADER}.")

    def _path_segments(self) -> list[str]:
        path = urllib.parse.urlsplit(self.path).path
        return [urllib.parse.unquote(segment) for segment in path.split("/") if segment]

    def _query(self) -> dict[str, str]:
        return {key: values[-1] for key, values in
                urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query).items()}

    def _route(self, method: str, segments: list[str]) -> None:
        match (method, segments):
            case ("GET", []):
                self._send_static("index.html")
            case ("GET", ["static", name]):
                self._send_static(name)
            case ("GET", ["api", "runs"]):
                self._send_json(HTTPStatus.OK, {"runs": list_runs(self.state.runs_root)})
            case ("POST", ["api", "runs"]):
                self._start_upload()
            case ("GET", ["api", "jobs", job_id]):
                self._send_job(job_id)
            case ("GET", ["api", "runs", run_id, "document"]):
                self._send_json(HTTPStatus.OK, self._document(run_id))
            case ("GET", ["api", "runs", run_id, "audio"]):
                self._send_audio(stored_audio(self._run_dir(run_id)))
            case ("GET", ["api", "runs", run_id, export]) if export in EXPORT_FORMATS:
                self._send_export(run_id, export)
            case ("PUT", ["api", "runs", run_id, "paragraphs", first_word, "text"]):
                self._edit_text(run_id, first_word)
            case ("POST", ["api", "runs", run_id, "reassign"]):
                self._reassign(run_id)
            case ("PUT", ["api", "runs", run_id, "speakers"]):
                self._rename_speakers(run_id)
            case _:
                raise HttpError(HTTPStatus.NOT_FOUND, f"No route for {method} {self.path}.")

    def _send_static(self, name: str) -> None:
        if name not in STATIC_CONTENT_TYPES:
            raise HttpError(HTTPStatus.NOT_FOUND, f"No static file {name}.")
        self._send_bytes(HTTPStatus.OK, static_file(name), STATIC_CONTENT_TYPES[name])

    def _start_upload(self) -> None:
        if not self.state.upload_lock.acquire(blocking=False):
            raise HttpError(HTTPStatus.CONFLICT, "Another upload is in progress.")
        try:
            if self.state.job_running():
                raise HttpError(HTTPStatus.CONFLICT, "A transcription is already running. Wait until it finishes.")
            extension = audio_extension(urllib.parse.unquote(self.headers.get(AUDIO_FILENAME_HEADER, "")))
            query = self._query()
            names = {"A": query.get("name_a", "").strip(), "B": query.get("name_b", "").strip()}
            if not all(names.values()):
                raise ValueError("Enter a name for both speakers.")
            length = self._content_length()
            run_dir = create_run_dir(self.state.runs_root)
            audio = run_dir / f"{STORED_AUDIO_STEM}{extension}"
            try:
                self._receive_body(audio, length)
            except BaseException:
                shutil.rmtree(run_dir)
                raise
            job = TranscriptionJob(run_dir, audio, names)
            self.state.jobs[job.job_id] = job
            job.start()
        finally:
            self.state.upload_lock.release()
        self._send_json(HTTPStatus.ACCEPTED, {"job_id": job.job_id, "run_id": run_dir.name})

    def _content_length(self) -> int:
        header = self.headers.get("Content-Length")
        if header is None:
            raise HttpError(HTTPStatus.LENGTH_REQUIRED, "The upload needs a Content-Length header.")
        length = int(header)
        if length <= 0:
            raise ValueError("The uploaded file is empty.")
        return length

    def _receive_body(self, target: Path, length: int) -> None:
        remaining = length
        with target.open("wb") as output:
            while remaining:
                chunk = self.rfile.read(min(UPLOAD_CHUNK_BYTES, remaining))
                if not chunk:
                    raise ValueError("The upload ended before the whole file was received.")
                output.write(chunk)
                remaining -= len(chunk)

    def _send_job(self, job_id: str) -> None:
        job = self.state.jobs.get(job_id)
        if job is None:
            raise HttpError(HTTPStatus.NOT_FOUND, f"Unknown job {job_id}.")
        status = job.status()
        self._send_json(HTTPStatus.OK, dict(state=status.state, phase=status.phase, detail=status.detail,
                                            error=status.error, run_id=status.run_id))

    def _run_dir(self, run_id: str) -> Path:
        return resolve_run_dir(self.state.runs_root, run_id)

    def _document(self, run_id: str) -> dict:
        run_dir = self._run_dir(run_id)
        stored_audio(run_dir)
        readable = load_readable(run_dir)
        document = build_editor_document(readable, load_corrected_paragraphs(run_dir, readable),
                                         load_speaker_names(run_dir, readable))
        document["run_id"] = run_id
        return document

    def _edit_text(self, run_id: str, first_word: str) -> None:
        text = self._json_body().get("text")
        if not isinstance(text, str):
            raise ValueError("The request needs a text string.")
        self._update_corrections(run_id, lambda readable, paragraphs, speakers: (
            edit_paragraph_text(paragraphs, int(first_word), text), speakers))

    def _reassign(self, run_id: str) -> None:
        body = self._json_body()
        first_word, last_word, speaker = body.get("first_word"), body.get("last_word"), body.get("speaker")
        if type(first_word) is not int or type(last_word) is not int or not isinstance(speaker, str):
            raise ValueError("The request needs integer first_word/last_word and a speaker label.")
        self._update_corrections(run_id, lambda readable, paragraphs, speakers: (
            reassign_words(paragraphs, readable_source_words(readable), set(readable["speakers"]),
                           first_word, last_word, speaker), speakers))

    def _rename_speakers(self, run_id: str) -> None:
        names = self._json_body()
        self._update_corrections(run_id, lambda readable, paragraphs, speakers: (paragraphs, names))

    def _update_corrections(self, run_id: str, change) -> None:
        run_dir = self._run_dir(run_id)
        with self.state.corrections_lock:
            readable = load_readable(run_dir)
            paragraphs, speakers = change(readable, load_corrected_paragraphs(run_dir, readable),
                                          load_speaker_names(run_dir, readable))
            save_corrections(run_dir, readable, speakers, paragraphs)
        self._send_json(HTTPStatus.OK, self._document(run_id))

    def _json_body(self) -> dict:
        body = json.loads(self.rfile.read(self._content_length()).decode("utf-8"))
        if not isinstance(body, dict):
            raise ValueError("The request body must be a JSON object.")
        return body

    def _send_audio(self, audio: Path) -> None:
        size = audio.stat().st_size
        content_type = AUDIO_CONTENT_TYPES[audio.suffix]
        range_header = self.headers.get("Range")
        if range_header is None:
            start, end, status = 0, size - 1, HTTPStatus.OK
        else:
            start, end = parse_byte_range(range_header, size)
            status = HTTPStatus.PARTIAL_CONTENT
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with audio.open("rb") as source:
            source.seek(start)
            remaining = end - start + 1
            while remaining:
                chunk = source.read(min(UPLOAD_CHUNK_BYTES, remaining))
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _send_export(self, run_id: str, export: str) -> None:
        suffix, markdown, content_type = EXPORT_FORMATS[export]
        text = corrected_text(self._document(run_id), markdown=markdown)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Disposition", f'attachment; filename="{run_id}_corrected.{suffix}"')
        body = text.encode("utf-8")
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: HTTPStatus, value: dict) -> None:
        self._send_bytes(status, json.dumps(value, ensure_ascii=False).encode("utf-8"),
                         "application/json; charset=utf-8")

    def _send_bytes(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def audio_extension(filename: str) -> str:
    """Return the lower-case audio extension of an upload's filename.

    Raises:
        ValueError: If the file is not WAV, MP3 or M4A.
    """
    extension = Path(filename).suffix.lower()
    if extension not in AUDIO_CONTENT_TYPES:
        raise ValueError(UNSUPPORTED_FORMAT_MESSAGE)
    return extension


def create_run_dir(runs_root: Path) -> Path:
    """Create a new, timestamp-named run directory under runs_root."""
    run_dir = runs_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    run_dir.mkdir(parents=True)
    return run_dir


def resolve_run_dir(runs_root: Path, run_id: str) -> Path:
    """Resolve a run id to its directory directly under runs_root.

    Raises:
        HttpError: If the id is not a single existing directory name under runs_root.
    """
    if run_id in {"", ".", ".."} or "/" in run_id or "\\" in run_id:
        raise HttpError(HTTPStatus.NOT_FOUND, f"Unknown run {run_id}.")
    run_dir = (runs_root / run_id).resolve()
    if run_dir.parent != runs_root.resolve() or not run_dir.is_dir():
        raise HttpError(HTTPStatus.NOT_FOUND, f"Unknown run {run_id}.")
    return run_dir


def stored_audio(run_dir: Path) -> Path:
    """Return the run's stored copy of the uploaded audio.

    Raises:
        HttpError: If the run directory holds no stored audio copy.
    """
    candidates = [run_dir / f"{STORED_AUDIO_STEM}{extension}" for extension in AUDIO_CONTENT_TYPES]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise HttpError(HTTPStatus.CONFLICT, "This run has no stored audio copy and cannot be opened in the "
                        "editor. Only runs started from the web UI can be edited.")
    return existing[0]


def load_readable(run_dir: Path) -> dict:
    """Load the run's readable (03) transcript.

    Raises:
        HttpError: If the run has no readable transcript yet.
    """
    path = run_dir / READABLE_FILENAME
    if not path.is_file():
        raise HttpError(HTTPStatus.NOT_FOUND, "This run has no readable transcript yet.")
    return json.loads(path.read_text(encoding="utf-8"))


def list_runs(runs_root: Path) -> list[dict]:
    """List runs with a readable transcript, newest first."""
    runs = [path for path in runs_root.iterdir() if path.is_dir() and (path / READABLE_FILENAME).is_file()]
    return [{"run_id": path.name} for path in sorted(runs, key=lambda path: path.name, reverse=True)]


def static_file(name: str) -> bytes:
    """Read a frontend file shipped as package data."""
    return resources.files(__package__).joinpath("static", name).read_bytes()
