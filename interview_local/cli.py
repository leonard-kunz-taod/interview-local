from __future__ import annotations
import argparse
from datetime import datetime
import getpass
import json
import os
from pathlib import Path
import platform
import sys
from .core import export, render
from .artifacts import save_readable, save_speakers, readable_text


def get_token():
    from .models import local_diarization_model
    if local_diarization_model():
        return None
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    from huggingface_hub import get_token as cached_token
    token = cached_token()
    if token:
        return token
    print("Accept the pyannote/speaker-diarization-community-1 conditions:")
    print("https://huggingface.co/pyannote/speaker-diarization-community-1")
    print("Create a read token: https://huggingface.co/settings/tokens")
    return getpass.getpass("HF token (hidden; this run only, Enter to skip): ").strip() or None


def doctor():
    print(f"System: {platform.system()} {platform.release()} / {platform.machine()}")
    print(f"Python: {platform.python_version()}")
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"RAM: {memory.total / 2**30:.1f} GiB total; {memory.available / 2**30:.1f} GiB available")
    except (ImportError, OSError):
        print("Could not read RAM information automatically.")
    import torch
    print(f"Apple Metal/MPS available: {torch.backends.mps.is_available()}")
    print(f"NVIDIA CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Dedicated GPU memory: {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        print("Apple Silicon: CPU and GPU share RAM; there is no additional dedicated VRAM.")
    print("Speech models: CPU INT8/Float32, sequential worker processes.")
    print("Readability: local Ollama / Ministral 3 8B, 8192-token context, after speech models exit.")
    from .llm import OllamaClient
    try:
        print("Local LLM:", OllamaClient().ready())
    except RuntimeError as exc:
        print(exc)


def new_output(path):
    path = path or Path("transcripts") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    path = path.expanduser().resolve()
    # Separate runs must never overwrite preserved stage artifacts.
    if path.exists() and any(path.iterdir()):
        raise ValueError("Output directory is not empty. Choose a new --output directory.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def main():
    args = argument_parser().parse_args()
    try:
        return run_command(args)
    except KeyboardInterrupt:
        print("\nProcessing cancelled. Completed stage files remain on disk.", file=sys.stderr)  # pragma: no mutate - message wording; the stream and exit code 130 are asserted
        return 130
    except Exception as exc:
        print(f"Error: {exc}\nCompleted earlier stages remain on disk, if any.", file=sys.stderr)
        return 1


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="German interviews: raw speech, speaker attribution and readable local-LLM drafts.")
    parser.add_argument("audio", nargs="?", type=Path)
    parser.add_argument("--output", type=Path, help="New, empty output directory")
    parser.add_argument("--gui", action="store_true", help="Optional native Tk window")
    parser.add_argument("--web", action="store_true", help="Local browser UI on 127.0.0.1 with transcript editor")
    parser.add_argument("--runs-dir", type=Path, default=Path("transcripts"),
                        help="Parent directory of web UI runs (default: transcripts/)")
    parser.add_argument("--doctor", action="store_true", help="Check hardware and the local model service")
    parser.add_argument("--prepare", action="store_true", help="Prepare the speech models")
    parser.add_argument("--prepare-llm", action="store_true", help="Download Ministral via local Ollama (no HF login)")
    parser.add_argument("--offline", action="store_true", help="No model downloads; use local caches")
    alignment = parser.add_mutually_exclusive_group()
    alignment.add_argument("--align", dest="alignment", action="store_true", help="Optional German WhisperX alignment")
    alignment.add_argument("--no-align", dest="alignment", action="store_false")  # pragma: no mutate - it stores False, the default, so under any other dest alignment stays False too
    parser.set_defaults(alignment=False)
    parser.add_argument("--name-a", default="Speaker A")
    parser.add_argument("--name-b", default="Speaker B")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--rename", type=Path, help="Rename an existing speaker transcript without inference")
    modes.add_argument("--enhance", type=Path, help="Create a readable draft from an existing speaker JSON")
    modes.add_argument("--review", type=Path, help="Separately compare a readable JSON against its embedded source")
    modes.add_argument("--raw-only", action="store_true", help="Speech-to-text only; no diarization, LLM or HF account")
    parser.add_argument("--no-llm", action="store_true", help="Stop after the preserved speaker transcript")
    return parser


def run_command(args: argparse.Namespace) -> int:
    if args.web:
        return serve_web(args.runs_dir)
    reject_audio_with_saved_stage(args)
    reject_offline_preparation(args)
    if args.doctor:
        doctor()
        return 0
    if args.gui:
        return launch_gui(args)
    return run_terminal_command(args)


def serve_web(runs_dir: Path) -> int:
    from .web.server import serve
    serve(runs_dir.expanduser().resolve())
    return 0


def reject_audio_with_saved_stage(args: argparse.Namespace) -> None:
    saved_stage_given = any((args.rename, args.enhance, args.review))
    if args.audio and saved_stage_given:
        raise ValueError("Supply audio or a saved-stage command, not both.")


def reject_offline_preparation(args: argparse.Namespace) -> None:
    preparation_requested = args.prepare or args.prepare_llm
    if args.offline and preparation_requested:
        raise ValueError("Preparation downloads models; do not combine it with --offline.")


def launch_gui(args: argparse.Namespace) -> int:
    """Open the Tk window; without Tk, continue in the terminal."""
    try:
        from .gui import launch
    except ImportError:
        print("Tk is not installed. Falling back to the terminal dialog.")
        return run_terminal_command(args)
    launch()
    return 0


def run_terminal_command(args: argparse.Namespace) -> int:
    if args.prepare or args.prepare_llm:
        return prepare_models(args)
    if args.review:
        return review_saved_draft(args)
    return run_transcript_command(args)


def run_transcript_command(args: argparse.Namespace) -> int:
    if args.enhance:
        return enhance_saved_transcript(args)
    if args.rename:
        return rename_saved_speakers(args)
    return transcribe_audio(args)


def prepare_models(args: argparse.Namespace) -> int:
    from .llm import OllamaClient
    from .pipeline import prepare
    if args.prepare:
        prepare(token=get_token(), alignment=args.alignment)
    if args.prepare_llm:
        OllamaClient().prepare()
    print("Preparation complete.")
    return 0


def review_saved_draft(args: argparse.Namespace) -> int:
    from .llm import review
    edited = json.loads(args.review.read_text(encoding="utf-8"))
    output = new_output(args.output)
    checked = review(edited)
    save_readable(checked, output)
    print(readable_text(checked))
    print(f"Review saved to {output}. Source and readable draft were not overwritten.")
    return 0


def enhance_saved_transcript(args: argparse.Namespace) -> int:
    from .llm import enhance
    source = json.loads(args.enhance.read_text(encoding="utf-8"))
    if "words" not in source or "speakers" not in source:
        raise ValueError("Enhancement requires a stage-2 speaker transcript JSON.")
    output = new_output(args.output)
    edited = enhance(source)
    save_readable(edited, output)
    print(readable_text(edited))
    print(f"Readable TXT/Markdown and audit JSON saved to {output}")
    return 0


def rename_saved_speakers(args: argparse.Namespace) -> int:
    result = json.loads(args.rename.read_text(encoding="utf-8"))
    if "speakers" not in result or "segments" not in result:
        raise ValueError("Renaming requires a speaker transcript JSON.")
    ask_speaker_names(args)
    output = new_output(args.output)
    save_named_speakers(result, args, output)
    print(render(result))
    print(f"Completed stages saved to {output}")
    return 0


def transcribe_audio(args: argparse.Namespace) -> int:
    audio = existing_audio(args.audio or ask_audio_path())
    if args.raw_only:
        return transcribe_raw_only(audio, args)
    return transcribe_speakers(audio, args)


def ask_audio_path() -> Path:
    print("Interview Local — German / two speakers / local processing only")
    print("Enter a path or drag a WAV, MP3 or M4A file into the terminal.")
    raw = input("Audio file: ").strip()
    if os.name == "nt":
        return Path(raw.strip('"'))
    import shlex
    paths = shlex.split(raw)
    return Path(paths[0]) if len(paths) == 1 else Path(raw)


def existing_audio(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_file():
        raise ValueError("Audio file not found. Enter its full path.")
    return path


def transcribe_raw_only(audio: Path, args: argparse.Namespace) -> int:
    from .pipeline import process
    output = new_output(args.output)
    process(audio, token=None, offline=args.offline, alignment=args.alignment, output_dir=output, raw_only=True)  # pragma: no mutate - raw-only processing returns before alignment and needs no token
    print((output / "01_raw.txt").read_text(encoding="utf-8"))
    print(f"Raw transcript saved to {output}")
    return 0


def transcribe_speakers(audio: Path, args: argparse.Namespace) -> int:
    """Transcribe with speaker attribution; names are asked when the audio path was asked as well."""
    from .llm import OllamaClient
    from .pipeline import process
    if not args.no_llm:
        OllamaClient().ready()
    output = new_output(args.output)
    token = None if args.offline else get_token()
    result = process(audio, token=token, offline=args.offline, alignment=args.alignment,
                     output_dir=output, raw_only=False)  # pragma: no mutate - None and the default are falsy like False
    if args.audio is None:
        ask_speaker_names(args)
    save_named_speakers(result, args, output)
    finish_transcript(result, args, output)
    return 0


def ask_speaker_names(args: argparse.Namespace) -> None:
    args.name_a = input("Name for Speaker A [Speaker A]: ").strip() or args.name_a
    args.name_b = input("Name for Speaker B [Speaker B]: ").strip() or args.name_b


def save_named_speakers(result: dict, args: argparse.Namespace, output: Path) -> None:
    result["speakers"].update(A=args.name_a, B=args.name_b)
    save_speakers(result, output)
    export(result, output)  # Legacy stage-2 filenames remain available.


def finish_transcript(result: dict, args: argparse.Namespace, output: Path) -> None:
    if args.no_llm:
        print(render(result))
    else:
        draft_readable(result, output)
    print(f"Completed stages saved to {output}")


def draft_readable(result: dict, output: Path) -> None:
    from .llm import enhance
    edited = enhance(result)
    save_readable(edited, output)
    print(readable_text(edited))
    print("For the separate comparison pass, run --review on 03_readable.json.")
