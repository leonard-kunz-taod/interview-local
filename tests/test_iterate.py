import ast
import hashlib
import json
import os
import sys
import tempfile
import tomllib
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from support.fake_ollama import DIGEST, FakeOllama, ScriptedModel, served_by

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


REFERENCE = "Beg\tEnd\tSpeaker\tText\n0\t1\tA\tGuten Tag\n1\t2\tB\tHallo Welt\n"
MISATTRIBUTED_STAGE2 = {
    "stage": "speakers",
    "speakers": {"A": "Speaker A", "B": "Speaker B"},
    "words": [
        {"word_id": 0, "start": 0, "end": 0.5, "text": "Guten", "speaker": "A"},
        {"word_id": 1, "start": 0.5, "end": 1, "text": "Tag", "speaker": "A"},
        {"word_id": 2, "start": 1, "end": 1.5, "text": "Hallo", "speaker": "A"},
        {"word_id": 3, "start": 1.5, "end": 2, "text": "Welt", "speaker": "B"},
    ],
    "diarization": [
        {"start": 0, "end": 1.5, "speaker": "A"},
        {"start": 1.5, "end": 2, "speaker": "B"},
    ],
}
CORRECTED_STAGE3 = {
    "stage": "readable",
    "source_transcript": MISATTRIBUTED_STAGE2,
    "paragraphs": [
        {"speaker": "A", "text": "Guten Tag", "source_word_ids": [0, 1]},
        {"speaker": "B", "text": "Hallo Welt", "source_word_ids": [2, 3]},
    ],
}


@contextmanager
def reference_case():
    """A case scored in [0, 2) against REFERENCE."""
    with tempfile.TemporaryDirectory() as directory:
        reference = Path(directory) / "sample.tsv"
        reference.write_text(REFERENCE, encoding="utf-8")
        yield iterate.BenchmarkCase(
            name="case",
            speakers=Path(directory) / "02_speakers.json",
            speakers_sha256=sha256(b"stage 2"),
            reference=reference,
            reference_sha256=sha256(REFERENCE.encode()),
            offset=0,
            start=0,
            end=2,
        )


class LocalRunStore(ModuleType):
    """The MLflow calls the benchmark makes, recording runs in memory."""

    def __init__(self):
        super().__init__("mlflow")
        self.runs = {}
        self.active = None
        self.tracking_uri = None
        self.experiments = {}
        self.experiment_id = None

    def set_tracking_uri(self, uri):
        self.tracking_uri = uri

    def get_experiment_by_name(self, name):
        return self.experiments.get(name)

    def create_experiment(self, name, artifact_location):
        experiment_id = str(len(self.experiments))
        self.experiments[name] = SimpleNamespace(
            experiment_id=experiment_id, artifact_location=artifact_location
        )
        return experiment_id

    def set_experiment(self, experiment_id):
        self.experiment_id = experiment_id

    @contextmanager
    def start_run(self, run_name):
        self.active = self.runs.setdefault(run_name, {})
        yield
        self.active = None

    def log_params(self, params):
        self.active["params"] = params

    def log_metrics(self, metrics):
        self.active["metrics"] = metrics

    def flush_trace_async_logging(self):
        pass


def paragraph(content_changed, speaker_changed):
    return {"content_changed": content_changed, "speaker_changed": speaker_changed}


class CachedInputTests(unittest.TestCase):
    def aborted(self, speakers, reference, speakers_sha256):
        """Ollama requests made and the abort message of a run on these inputs."""
        with tempfile.TemporaryDirectory() as directory:
            if speakers is not None:
                (Path(directory) / "02_speakers.json").write_bytes(speakers)
            if reference is not None:
                (Path(directory) / "sample.tsv").write_bytes(reference)
            cases_file = Path(directory) / "cases.json"
            cases_file.write_text(json.dumps([case_entry(directory, speakers_sha256)]))
            service = FakeOllama()
            with (
                patch.object(iterate, "CASES_FILE", cases_file),
                served_by(service),
                self.assertRaises(SystemExit) as abort,
            ):
                iterate.main()
        return service.requests, str(abort.exception.code)

    def test_unusable_cached_input_aborts_before_any_ollama_request(self):
        content = json.dumps(STAGE2).encode()
        unusable = {
            "changed stage 2": (content, b"reference", sha256(b"other")),
            "missing stage 2": (None, b"reference", sha256(content)),
            "changed reference": (content, b"other", sha256(content)),
            "missing reference": (content, None, sha256(content)),
        }
        requests = {name: self.aborted(*inputs)[0] for name, inputs in unusable.items()}
        self.assertEqual(requests, {name: [] for name in unusable})

    def test_changed_stage_two_abort_names_expected_and_found_sha256(self):
        content = json.dumps(STAGE2).encode()
        expected = sha256(b"the benchmarked stage 2")
        _, message = self.aborted(content, b"reference", expected)
        self.assertEqual(
            (expected in message, sha256(content) in message), (True, True)
        )

    def test_missing_stage_two_abort_names_the_expected_sha256(self):
        expected = sha256(b"the benchmarked stage 2")
        _, message = self.aborted(None, b"reference", expected)
        self.assertIn(expected, message)

    def test_missing_reference_names_the_fetch_command(self):
        content = json.dumps(STAGE2).encode()
        _, message = self.aborted(content, None, sha256(content))
        self.assertIn(iterate.FETCH_REFERENCE, message)

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

    def test_stages_are_compared_by_wer_and_cpwer_with_deltas(self):
        with reference_case() as case:
            comparison = iterate.stage_comparison(
                case, MISATTRIBUTED_STAGE2, CORRECTED_STAGE3
            )
        self.assertEqual(
            comparison,
            {
                "wer_stage2": 0.0,
                "wer_stage3": 0.0,
                "wer_delta": 0.0,
                "cpwer_stage2": 0.5,
                "cpwer_stage3": 0.0,
                "cpwer_delta": -0.5,
            },
        )

    def test_case_is_logged_as_one_run_with_its_parameters_and_metrics(self):
        store = LocalRunStore()
        with (
            reference_case() as case,
            served_by(FakeOllama(ScriptedModel(invalid_edits=2))),
            patch.object(iterate.time, "perf_counter", side_effect=[10.0, 12.5]),
            patch("builtins.print"),
        ):
            iterate._benchmark_run(store, case, MISATTRIBUTED_STAGE2, DIGEST)
            expected_params = iterate.run_params(case, DIGEST)
        self.assertEqual(
            store.runs,
            {
                "case": {
                    "params": expected_params,
                    "metrics": {
                        "wer_stage2": 0.0,
                        "wer_stage3": 0.0,
                        "wer_delta": 0.0,
                        "cpwer_stage2": 0.5,
                        "cpwer_stage3": 0.5,
                        "cpwer_delta": 0.0,
                        "rejected_chunks": 1,
                        "changed_paragraphs": 0,
                        "llm_stage_seconds": 2.5,
                    },
                }
            },
        )

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
        params = iterate.run_params(case, DIGEST)
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
                "model_digest": DIGEST,
            },
        )


@contextmanager
def local_tracking(store):
    """The store stands in for mlflow, tracking into a temporary .mlflow/."""
    with (
        tempfile.TemporaryDirectory() as directory,
        patch.dict(sys.modules, {"mlflow": store}),
        patch.dict(os.environ),
        patch.object(iterate, "MLFLOW_DIR", Path(directory) / ".mlflow"),
    ):
        yield iterate.MLFLOW_DIR


class LocalTrackingTests(unittest.TestCase):
    def test_runs_are_tracked_in_sqlite_inside_the_mlflow_directory(self):
        store = LocalRunStore()
        with local_tracking(store) as mlflow_dir:
            iterate._local_mlflow()
            created = mlflow_dir.is_dir()
        self.assertEqual(
            (store.tracking_uri, created),
            (f"sqlite:///{(mlflow_dir / 'mlflow.db').as_posix()}", True),
        )

    def test_missing_experiment_is_created_with_artifacts_in_the_mlflow_directory(
        self,
    ):
        store = LocalRunStore()
        with local_tracking(store) as mlflow_dir:
            iterate._local_mlflow()
        experiment = store.experiments[iterate.EXPERIMENT_NAME]
        self.assertEqual(
            (experiment.artifact_location, store.experiment_id),
            ((mlflow_dir / "artifacts").as_posix(), experiment.experiment_id),
        )

    def test_existing_experiment_is_reused(self):
        store = LocalRunStore()
        existing = SimpleNamespace(experiment_id="7", artifact_location="kept")
        store.experiments[iterate.EXPERIMENT_NAME] = existing
        with local_tracking(store):
            iterate._local_mlflow()
        self.assertEqual(
            (store.experiments, store.experiment_id),
            ({iterate.EXPERIMENT_NAME: existing}, "7"),
        )

    def test_tracking_sends_no_telemetry_or_agent_hints(self):
        with local_tracking(LocalRunStore()):
            iterate._local_mlflow()
            switches = (
                os.environ["MLFLOW_DISABLE_TELEMETRY"],
                os.environ["MLFLOW_DISABLE_AGENT_HINT"],
            )
        self.assertEqual(switches, ("true", "1"))

    def test_the_configured_mlflow_is_returned(self):
        store = LocalRunStore()
        with local_tracking(store):
            configured = iterate._local_mlflow()
        self.assertIs(configured, store)


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
    nodes = list(ast.walk(ast.parse(source.read_text("utf-8"))))
    return import_statement_modules(nodes) + from_import_modules(nodes)


def import_statement_modules(nodes):
    """Modules named by ``import a, b`` statements."""
    return [
        alias.name
        for node in nodes
        if isinstance(node, ast.Import)
        for alias in node.names
    ]


def from_import_modules(nodes):
    """Modules named by ``from a import b``; ``from . import b`` names none."""
    return [
        node.module
        for node in nodes
        if isinstance(node, ast.ImportFrom) and node.module
    ]


if __name__ == "__main__":
    unittest.main()
