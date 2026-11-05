"""Score the LLM readability edit (stage 3) on cached stage-2 transcripts.

Runs only the LLM stage, so prompt and constant changes can be compared without
repeating transcription and speaker recognition. Each case becomes one local
MLflow run in .mlflow/, and each LLM call a trace.

    uv run --group eval python benchmark/iterate.py
"""

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# The benchmark writes no files outside .mlflow/, including bytecode caches of
# the modules imported below.
sys.dont_write_bytecode = True

from interview_local import evaluate, llm

REPO_ROOT = Path(__file__).resolve().parent.parent
CASES_FILE = REPO_ROOT / "benchmark" / "cases.json"
MLFLOW_DIR = REPO_ROOT / ".mlflow"
EXPERIMENT_NAME = "llm-readable-edit"
FETCH_REFERENCE = "python tools/fetch_test_sample.py test_data/callhome"
LABEL_WIDTH = 20
COLUMN_WIDTH = 10


@dataclass(frozen=True)
class BenchmarkCase:
    """A cached stage-2 transcript and the reference window it is scored in."""

    name: str
    speakers: Path
    speakers_sha256: str
    reference: Path
    reference_sha256: str
    offset: float
    start: float
    end: float


def main() -> None:
    cases = load_cases(CASES_FILE)
    inputs = []
    for case in cases:
        speakers = verified_speakers(case)
        verify_reference(case)
        inputs.append((case, speakers))
    identity = llm.OllamaClient().ready()
    mlflow = _local_mlflow()
    _trace_llm_calls(mlflow)
    for case, speakers in inputs:
        print(format_table(_benchmark_run(mlflow, case, speakers, identity)))


def load_cases(cases_file: Path) -> list[BenchmarkCase]:
    """Benchmark cases, with their file paths resolved against the repository root."""
    entries = json.loads(cases_file.read_text(encoding="utf-8"))
    return [
        BenchmarkCase(
            name=entry["name"],
            speakers=REPO_ROOT / entry["speakers"],
            speakers_sha256=entry["speakers_sha256"],
            reference=REPO_ROOT / entry["reference"],
            reference_sha256=entry["reference_sha256"],
            offset=entry["offset"],
            start=entry["start"],
            end=entry["end"],
        )
        for entry in entries
    ]


def verified_speakers(case: BenchmarkCase) -> dict:
    """The cached stage-2 transcript, if it is byte-identical to the benchmarked one.

    Raises:
        SystemExit: The file is missing or its SHA-256 differs.
    """
    if not case.speakers.is_file():
        raise SystemExit(
            f"Cached stage 2 not found: {case.speakers} "
            f"(expected SHA-256 {case.speakers_sha256})."
        )
    content = _checked_bytes(case.speakers, case.speakers_sha256)
    return json.loads(content.decode("utf-8"))


def verify_reference(case: BenchmarkCase) -> None:
    """Check that the reference TSV is present and byte-identical to the expected one.

    Raises:
        SystemExit: The file is missing or its SHA-256 differs.
    """
    if not case.reference.is_file():
        raise SystemExit(
            f"Reference not found: {case.reference}. Fetch it with: {FETCH_REFERENCE}"
        )
    _checked_bytes(case.reference, case.reference_sha256)


def _checked_bytes(path: Path, expected_sha256: str) -> bytes:
    content = path.read_bytes()
    actual_sha256 = hashlib.sha256(content).hexdigest()
    if actual_sha256 != expected_sha256:
        raise SystemExit(
            f"{path} is not the benchmarked file: expected SHA-256 "
            f"{expected_sha256}, found {actual_sha256}. No LLM call was made."
        )
    return content


def changed_paragraph_count(readable: dict) -> int:
    """Paragraphs whose wording or speaker the LLM changed."""
    return sum(
        1
        for paragraph in readable["paragraphs"]
        if paragraph["content_changed"] or paragraph["speaker_changed"]
    )


def stage_comparison(
    case: BenchmarkCase, speakers: dict, readable: dict
) -> dict[str, float]:
    """WER and cpWER of stage 2 and stage 3 in the case's window, with deltas."""
    stage2 = _scored(case, speakers)
    stage3 = _scored(case, readable)
    return {
        "wer_stage2": stage2["wer"],
        "wer_stage3": stage3["wer"],
        "wer_delta": stage3["wer"] - stage2["wer"],
        "cpwer_stage2": stage2["cpwer"],
        "cpwer_stage3": stage3["cpwer"],
        "cpwer_delta": stage3["cpwer"] - stage2["cpwer"],
    }


def _scored(case: BenchmarkCase, transcript: dict) -> dict:
    return evaluate.score_transcript(
        case.reference, transcript, case.offset, case.start, case.end
    )


def run_params(case: BenchmarkCase, identity: dict) -> dict:
    """Everything that determines the LLM stage's output for one case."""
    return {
        "case": case.name,
        "speakers_sha256": case.speakers_sha256,
        "prompt_sha256": hashlib.sha256(llm.EDIT_PROMPT.encode("utf-8")).hexdigest(),
        "schema_sha256": llm.fingerprint(llm.MODEL_SCHEMA),
        "temperature": llm.TEMPERATURE,
        "similarity_threshold": llm.MIN_WORDING_SIMILARITY,
        "chunk_words": llm.EDIT_CHUNK_WORDS,
        "model": llm.MODEL,
        "model_digest": identity["digest"],
    }


def format_table(metrics: dict[str, float]) -> str:
    """The run's metrics as a plain-text table."""
    rows = [
        _row("", "stage 2", "stage 3", "delta"),
        _rate_row("WER", metrics, "wer"),
        _rate_row("cpWER", metrics, "cpwer"),
        _row("rejected chunks", f"{metrics['rejected_chunks']}"),
        _row("changed paragraphs", f"{metrics['changed_paragraphs']}"),
        _row("LLM runtime (s)", f"{metrics['llm_stage_seconds']:.1f}"),
    ]
    return "\n".join(rows)


def _rate_row(label: str, metrics: dict[str, float], metric: str) -> str:
    return _row(
        label,
        f"{metrics[f'{metric}_stage2']:.4f}",
        f"{metrics[f'{metric}_stage3']:.4f}",
        f"{metrics[f'{metric}_delta']:+.4f}",
    )


def _row(label: str, *cells: str) -> str:
    return f"{label:<{LABEL_WIDTH}}" + "".join(
        f"{cell:>{COLUMN_WIDTH}}" for cell in cells
    )


def _benchmark_run(
    mlflow, case: BenchmarkCase, speakers: dict, identity: dict
) -> dict[str, float]:
    """Run the LLM stage on one case as an MLflow run; return its logged metrics."""
    with mlflow.start_run(run_name=case.name):
        mlflow.log_params(run_params(case, identity))
        started = time.perf_counter()
        readable = llm.enhance(speakers, print)
        llm_stage_seconds = time.perf_counter() - started
        metrics = {
            **stage_comparison(case, speakers, readable),
            "rejected_chunks": readable["metadata"]["rejected_chunks"],
            "changed_paragraphs": changed_paragraph_count(readable),
            "llm_stage_seconds": llm_stage_seconds,
        }
        mlflow.log_metrics(metrics)
        mlflow.flush_trace_async_logging()
    return metrics


def _local_mlflow():
    """MLflow tracking into .mlflow/ (sqlite), without telemetry."""
    os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    import mlflow

    MLFLOW_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(MLFLOW_DIR / 'mlflow.db').as_posix()}")
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            EXPERIMENT_NAME, artifact_location=(MLFLOW_DIR / "artifacts").as_posix()
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    return mlflow


def _trace_llm_calls(mlflow) -> None:
    """Record every chat, edit attempt and wording check as an MLflow span.

    A rejected edit attempt is recorded as an error span carrying the rejection
    reason. Replaces the functions in ``llm``; a renamed one raises AttributeError.
    """
    llm.OllamaClient.chat = mlflow.trace(llm.OllamaClient.chat, span_type="LLM")
    llm._edit_attempt = mlflow.trace(llm._edit_attempt)
    llm._similarity = mlflow.trace(llm._similarity)


if __name__ == "__main__":
    main()
