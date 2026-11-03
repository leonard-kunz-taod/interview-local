from __future__ import annotations

import argparse
import getpass
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

from . import llm, pipeline
from .artifacts import readable_text, save_readable, save_speakers_with_aliases
from .core import render


def main() -> int:
    """Run the command line and return the process exit code."""
    args = _parser().parse_args()
    try:
        return _run(args)
    except KeyboardInterrupt:
        print(
            "\nProcessing cancelled. Completed stage files remain on disk.",
            file=sys.stderr,
        )
        return 130
    except Exception as exc:
        print(
            f"Error: {exc}\nCompleted earlier stages remain on disk, if any.",
            file=sys.stderr,
        )
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="German interviews: raw speech, speaker attribution "
        "and readable local-LLM drafts."
    )
    parser.add_argument("audio", nargs="?", type=Path)
    parser.add_argument("--output", type=Path, help="New, empty output directory")
    parser.add_argument("--gui", action="store_true", help="Optional native Tk window")
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Check hardware and the local model service",
    )
    parser.add_argument(
        "--prepare", action="store_true", help="Prepare the speech models"
    )
    parser.add_argument(
        "--prepare-llm",
        action="store_true",
        help="Download Ministral via local Ollama (no HF login)",
    )
    parser.add_argument(
        "--offline", action="store_true", help="No model downloads; use local caches"
    )
    alignment = parser.add_mutually_exclusive_group()
    alignment.add_argument(
        "--align",
        dest="alignment",
        action="store_true",
        help="Optional German WhisperX alignment",
    )
    alignment.add_argument("--no-align", dest="alignment", action="store_false")
    parser.set_defaults(alignment=False)
    parser.add_argument("--name-a", default="Speaker A")
    parser.add_argument("--name-b", default="Speaker B")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--rename",
        type=Path,
        help="Rename an existing speaker transcript without inference",
    )
    modes.add_argument(
        "--enhance",
        type=Path,
        help="Create a readable draft from an existing speaker JSON",
    )
    modes.add_argument(
        "--review",
        type=Path,
        help="Separately compare a readable JSON against its embedded source",
    )
    modes.add_argument(
        "--raw-only",
        action="store_true",
        help="Speech-to-text only; no diarization, LLM or HF account",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Stop after the preserved speaker transcript",
    )
    return parser


def _run(args: argparse.Namespace) -> int:
    _reject_audio_with_saved_stage(args)
    _reject_offline_preparation(args)
    if args.doctor:
        _doctor()
        return 0
    if args.gui:
        try:
            from .gui import launch

            launch()
            return 0
        except ImportError:
            print("Tk is not installed. Falling back to the terminal dialog.")
    return _run_terminal_mode(args)


def _reject_audio_with_saved_stage(args: argparse.Namespace) -> None:
    if args.audio and _saved_stage_requested(args):
        raise ValueError("Supply audio or a saved-stage command, not both.")


def _reject_offline_preparation(args: argparse.Namespace) -> None:
    if args.offline and _preparation_requested(args):
        raise ValueError(
            "Preparation downloads models; do not combine it with --offline."
        )


def _saved_stage_requested(args: argparse.Namespace) -> bool:
    return bool(args.rename or args.enhance or args.review)


def _preparation_requested(args: argparse.Namespace) -> bool:
    return args.prepare or args.prepare_llm


def _run_terminal_mode(args: argparse.Namespace) -> int:
    if _preparation_requested(args):
        return _prepare(args.prepare, args.prepare_llm, args.alignment)
    if _saved_stage_requested(args):
        return _run_saved_stage(args)
    return _process_audio(args)


def _run_saved_stage(args: argparse.Namespace) -> int:
    if args.review:
        return _review_saved(args.review, args.output)
    if args.enhance:
        return _enhance_saved(args.enhance, args.output)
    return _rename_saved(args.rename, args.output, args.name_a, args.name_b)


def _doctor() -> None:
    print(f"System: {platform.system()} {platform.release()} / {platform.machine()}")
    print(f"Python: {platform.python_version()}")
    _print_ram()
    _print_accelerators()
    print("Speech models: CPU INT8/Float32, sequential worker processes.")
    print(
        "Readability: local Ollama / Ministral 3 8B, 8192-token context, "
        "after speech models exit."
    )
    _print_local_llm()


def _print_ram() -> None:
    try:
        import psutil

        memory = psutil.virtual_memory()
        print(
            f"RAM: {memory.total / 2**30:.1f} GiB total; "
            f"{memory.available / 2**30:.1f} GiB available"
        )
    except (ImportError, OSError):
        print("Could not read RAM information automatically.")


def _print_accelerators() -> None:
    import torch

    print(f"Apple Metal/MPS available: {torch.backends.mps.is_available()}")
    print(f"NVIDIA CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f"Dedicated GPU memory: {gpu_memory:.1f} GiB")
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        print(
            "Apple Silicon: CPU and GPU share RAM; there is no additional dedicated VRAM."
        )


def _print_local_llm() -> None:
    try:
        print("Local LLM:", llm.OllamaClient().ready())
    except RuntimeError as exc:
        print(exc)


def _prepare(speech_models: bool, language_model: bool, alignment: bool) -> int:
    if speech_models:
        pipeline.prepare(token=_get_token(), alignment=alignment)
    if language_model:
        llm.OllamaClient().prepare()
    print("Preparation complete.")
    return 0


def _review_saved(readable_file: Path, output_path: Path | None) -> int:
    edited = _load_stage(readable_file)
    output = _create_output(output_path)
    _save_and_print_readable(llm.review(edited), output)
    print(f"Review saved to {output}. Source and readable draft were not overwritten.")
    return 0


def _enhance_saved(speaker_file: Path, output_path: Path | None) -> int:
    source = _load_stage(speaker_file)
    if "words" not in source or "speakers" not in source:
        raise ValueError("Enhancement requires a stage-2 speaker transcript JSON.")
    output = _create_output(output_path)
    _save_and_print_readable(llm.enhance(source), output)
    print(f"Readable TXT/Markdown and audit JSON saved to {output}")
    return 0


def _rename_saved(
    speaker_file: Path, output_path: Path | None, name_a: str, name_b: str
) -> int:
    result = _load_stage(speaker_file)
    if "speakers" not in result or "segments" not in result:
        raise ValueError("Renaming requires a speaker transcript JSON.")
    names = _ask_speaker_names(name_a, name_b)
    output = _create_output(output_path)
    _save_named_speakers(result, names, output)
    print(render(result))
    print(f"Completed stages saved to {output}")
    return 0


def _process_audio(args: argparse.Namespace) -> int:
    audio = _audio_path(args)
    if args.raw_only:
        return _transcribe_raw(audio, args.offline, args.output)
    if not args.no_llm:
        llm.OllamaClient().ready()
    output = _create_output(args.output)
    result = _speaker_transcript(audio, args, output)
    _save_named_speakers(result, _speaker_names(args), output)
    if args.no_llm:
        print(render(result))
    else:
        _save_and_print_readable(llm.enhance(result), output)
        print("For the separate comparison pass, run --review on 03_readable.json.")
    print(f"Completed stages saved to {output}")
    return 0


def _audio_path(args: argparse.Namespace) -> Path:
    """The audio file from the command line, or asked for on the terminal."""
    audio = (args.audio or _ask_audio_path()).expanduser()
    if not audio.is_file():
        raise ValueError("Audio file not found. Enter its full path.")
    return audio


def _speaker_transcript(audio: Path, args: argparse.Namespace, output: Path) -> dict:
    """Run the speech stages; offline runs never ask for a Hub token."""
    token = None if args.offline else _get_token()
    return pipeline.process(
        audio,
        token=token,
        offline=args.offline,
        alignment=args.alignment,
        output_dir=output,
    )


def _speaker_names(args: argparse.Namespace) -> dict:
    """Names from the command line; asked for when the audio path was too."""
    if args.audio:
        return {"A": args.name_a, "B": args.name_b}
    return _ask_speaker_names(args.name_a, args.name_b)


def _transcribe_raw(audio: Path, offline: bool, output_path: Path | None) -> int:
    output = _create_output(output_path)
    pipeline.transcribe(audio, print, offline, output)
    print((output / "01_raw.txt").read_text(encoding="utf-8"))
    print(f"Raw transcript saved to {output}")
    return 0


def _get_token() -> str | None:
    """The HF token for community-1; None when a local copy or no token is used.

    Prompts on the terminal when neither the environment nor the HF cache has one.
    """
    from .models import local_diarization_model

    if local_diarization_model():
        return None
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    from huggingface_hub import get_token as cached_token

    token = cached_token()
    if token:
        return token
    return _ask_token()


def _ask_token() -> str | None:
    """A token entered for this run only; None when the entry is skipped."""
    print("Accept the pyannote/speaker-diarization-community-1 conditions:")
    print("https://huggingface.co/pyannote/speaker-diarization-community-1")
    print("Create a read token: https://huggingface.co/settings/tokens")
    prompt = "HF token (hidden; this run only, Enter to skip): "
    return getpass.getpass(prompt).strip() or None


def _ask_audio_path() -> Path:
    print("Interview Local — German / two speakers / local processing only")
    print("Enter a path or drag a WAV, MP3 or M4A file into the terminal.")
    entered = input("Audio file: ").strip()
    if os.name == "nt":
        return Path(entered.strip('"'))
    import shlex

    paths = shlex.split(entered)
    return Path(paths[0]) if len(paths) == 1 else Path(entered)


def _ask_speaker_names(name_a: str, name_b: str) -> dict:
    return {
        "A": input("Name for Speaker A [Speaker A]: ").strip() or name_a,
        "B": input("Name for Speaker B [Speaker B]: ").strip() or name_b,
    }


def _create_output(path: Path | None) -> Path:
    """Create a new, empty output directory; default to a timestamped one."""
    path = path or Path("transcripts") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    path = path.expanduser().resolve()
    # Separate runs must never overwrite preserved stage artifacts.
    if path.exists() and any(path.iterdir()):
        raise ValueError(
            "Output directory is not empty. Choose a new --output directory."
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_stage(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_named_speakers(result: dict, names: dict, output: Path) -> None:
    result["speakers"].update(names)
    save_speakers_with_aliases(result, output)


def _save_and_print_readable(result: dict, output: Path) -> None:
    save_readable(result, output)
    print(readable_text(result))
