import ast
import hashlib
import json
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

from support.fake_ollama import IDENTITY, FakeOllama, served_by

from benchmark import iterate
from interview_local import llm

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE2 = {"stage": "speakers", "speakers": {"A": "Speaker A"}, "words": []}
METRICS = {
    "wer_stage2": 0.2544,
    "wer_stage3": 0.25,
    "wer_delta": -0.0044,
    "cpwer_stage2": 0.3757,
    "cpwer_stage3": 0.4,
    "cpwer_delta": 0.0243,
    "rejected_chunks": 2,
    "changed_paragraphs": 5,
    "llm_stage_seconds": 123.45,
}


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def case_entry(directory, speakers_sha256):
    return {
        "name": "case",
        "speakers": str(Path(directory) / "02_speakers.json"),
        "speakers_sha256": speakers_sha256,
        "reference": str(Path(directory) / "sample.tsv"),
        "reference_sha256": sha256(b"reference"),
        "offset": 0,
        "start": 0,
        "end": 1,
    }


def paragraph(content_changed, speaker_changed):
    return {"content_changed": content_changed, "speaker_changed": speaker_changed}


class CachedStageTwoTests(unittest.TestCase):
    def test_changed_stage_two_aborts_before_any_ollama_request(self):
        content = json.dumps(STAGE2).encode()
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "02_speakers.json").write_bytes(content)
            (Path(directory) / "sample.tsv").write_bytes(b"reference")
            cases_file = Path(directory) / "cases.json"
            expected = sha256(b"the benchmarked stage 2")
            cases_file.write_text(json.dumps([case_entry(directory, expected)]))
            service = FakeOllama()
            with (
                patch.object(iterate, "CASES_FILE", cases_file),
                served_by(service),
                self.assertRaises(SystemExit) as abort,
            ):
                iterate.main()
        message = str(abort.exception.code)
        self.assertEqual(
            (service.requests, expected in message, sha256(content) in message),
            ([], True, True),
        )

    def test_missing_reference_names_the_fetch_command(self):
        content = json.dumps(STAGE2).encode()
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "02_speakers.json").write_bytes(content)
            cases_file = Path(directory) / "cases.json"
            cases_file.write_text(json.dumps([case_entry(directory, sha256(content))]))
            service = FakeOllama()
            with (
                patch.object(iterate, "CASES_FILE", cases_file),
                served_by(service),
                self.assertRaises(SystemExit) as abort,
            ):
                iterate.main()
        self.assertEqual(
            (service.requests, iterate.FETCH_REFERENCE in str(abort.exception.code)),
            ([], True),
        )

    def test_committed_case_paths_are_read_from_the_repository_root(self):
        case = iterate.load_cases(REPO_ROOT / "benchmark" / "cases.json")[0]
        self.assertEqual(
            (case.speakers, case.reference),
            (
                REPO_ROOT / "test_data" / "callhome" / "02_speakers.json",
                REPO_ROOT / "test_data" / "callhome" / "sample.tsv",
            ),
        )


class RunSummaryTests(unittest.TestCase):
    def test_paragraphs_with_changed_wording_or_speaker_are_counted(self):
        readable = {
            "paragraphs": [
                paragraph(content_changed=True, speaker_changed=False),
                paragraph(content_changed=False, speaker_changed=True),
                paragraph(content_changed=True, speaker_changed=True),
                paragraph(content_changed=False, speaker_changed=False),
            ]
        }
        self.assertEqual(iterate.changed_paragraph_count(readable), 3)

    def test_table_shows_both_stages_with_delta_and_llm_figures(self):
        rows = [row.split() for row in iterate.format_table(METRICS).splitlines()]
        self.assertEqual(
            rows,
            [
                ["stage", "2", "stage", "3", "delta"],
                ["WER", "0.2544", "0.2500", "-0.0044"],
                ["cpWER", "0.3757", "0.4000", "+0.0243"],
                ["rejected", "chunks", "2"],
                ["changed", "paragraphs", "5"],
                ["LLM", "runtime", "(s)", "123.5"],
            ],
        )

    def test_run_parameters_identify_prompt_decoding_and_model(self):
        case = iterate.load_cases(REPO_ROOT / "benchmark" / "cases.json")[0]
        params = iterate.run_params(case, IDENTITY)
        self.assertEqual(
            params,
            {
                "case": "callhome-de-sample",
                "speakers_sha256": case.speakers_sha256,
                "prompt_sha256": sha256(llm.EDIT_PROMPT.encode("utf-8")),
                "schema_sha256": llm.fingerprint(llm.MODEL_SCHEMA),
                "temperature": 0,
                "similarity_threshold": llm.MIN_WORDING_SIMILARITY,
                "chunk_words": llm.EDIT_CHUNK_WORDS,
                "model": llm.MODEL,
                "model_digest": IDENTITY["digest"],
            },
        )


class AppIsolationTests(unittest.TestCase):
    def test_app_modules_do_not_import_mlflow(self):
        imported = {
            name.split(".")[0]
            for source in (REPO_ROOT / "interview_local").rglob("*.py")
            for name in imported_modules(source)
        }
        self.assertNotIn("mlflow", imported)

    def test_mlflow_is_installed_only_with_the_eval_group(self):
        project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text("utf-8"))
        default_groups = project.get("tool", {}).get("uv", {}).get("default-groups")
        self.assertEqual(
            (
                any("mlflow" in d for d in project["project"]["dependencies"]),
                any("mlflow" in d for d in project["dependency-groups"]["eval"]),
                default_groups is None or "eval" not in default_groups,
            ),
            (False, True, True),
        )


def imported_modules(source):
    tree = ast.parse(source.read_text("utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


if __name__ == "__main__":
    unittest.main()
