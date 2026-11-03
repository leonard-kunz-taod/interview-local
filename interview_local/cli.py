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
    parser = argparse.ArgumentParser(description="German interviews: raw speech, speaker attribution and readable local-LLM drafts.")
    parser.add_argument("audio", nargs="?", type=Path)
    parser.add_argument("--output", type=Path, help="New, empty output directory")
    parser.add_argument("--gui", action="store_true", help="Optional native Tk window")
    parser.add_argument("--doctor", action="store_true", help="Check hardware and the local model service")
    parser.add_argument("--prepare", action="store_true", help="Prepare the speech models")
    parser.add_argument("--prepare-llm", action="store_true", help="Download Ministral via local Ollama (no HF login)")
    parser.add_argument("--offline", action="store_true", help="No model downloads; use local caches")
    alignment = parser.add_mutually_exclusive_group()
    alignment.add_argument("--align", dest="alignment", action="store_true", help="Optional German WhisperX alignment")
    alignment.add_argument("--no-align", dest="alignment", action="store_false")
    parser.set_defaults(alignment=False)
    parser.add_argument("--name-a", default="Speaker A")
    parser.add_argument("--name-b", default="Speaker B")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--rename", type=Path, help="Rename an existing speaker transcript without inference")
    modes.add_argument("--enhance", type=Path, help="Create a readable draft from an existing speaker JSON")
    modes.add_argument("--review", type=Path, help="Separately compare a readable JSON against its embedded source")
    modes.add_argument("--raw-only", action="store_true", help="Speech-to-text only; no diarization, LLM or HF account")
    parser.add_argument("--no-llm", action="store_true", help="Stop after the preserved speaker transcript")
    args = parser.parse_args()
    try:
        if args.audio and (args.rename or args.enhance or args.review):
            raise ValueError("Supply audio or a saved-stage command, not both.")
        if args.offline and (args.prepare or args.prepare_llm):
            raise ValueError("Preparation downloads models; do not combine it with --offline.")
        if args.doctor:
            doctor()
            return 0
        if args.gui:
            try:
                from .gui import launch
                launch()
                return 0
            except ImportError:
                print("Tk is not installed. Falling back to the terminal dialog.")
        from .pipeline import process, prepare
        from .llm import OllamaClient, enhance, review
        if args.prepare:
            prepare(token=get_token(), alignment=args.alignment)
        if args.prepare_llm:
            OllamaClient().prepare()
        if args.prepare or args.prepare_llm:
            print("Preparation complete.")
            return 0
        if args.review:
            edited = json.loads(args.review.read_text(encoding="utf-8"))
            output = new_output(args.output)
            checked = review(edited)
            save_readable(checked, output)
            print(readable_text(checked))
            print(f"Review saved to {output}. Source and readable draft were not overwritten.")
            return 0
        if args.enhance:
            source = json.loads(args.enhance.read_text(encoding="utf-8"))
            if "words" not in source or "speakers" not in source:
                raise ValueError("Enhancement requires a stage-2 speaker transcript JSON.")
            output = new_output(args.output)
            edited = enhance(source)
            save_readable(edited, output)
            print(readable_text(edited))
            print(f"Readable TXT/Markdown and audit JSON saved to {output}")
            return 0
        interactive = not args.audio and not args.rename
        if args.rename:
            result = json.loads(args.rename.read_text(encoding="utf-8"))
            if "speakers" not in result or "segments" not in result:
                raise ValueError("Renaming requires a speaker transcript JSON.")
        else:
            if not args.audio:
                print("Interview Local — German / two speakers / local processing only")
                print("Enter a path or drag a WAV, MP3 or M4A file into the terminal.")
                raw = input("Audio file: ").strip()
                if os.name == "nt":
                    args.audio = Path(raw.strip('"'))
                else:
                    import shlex
                    paths = shlex.split(raw)
                    args.audio = Path(paths[0]) if len(paths) == 1 else Path(raw)
            args.audio = args.audio.expanduser()
            if not args.audio.is_file():
                raise ValueError("Audio file not found. Enter its full path.")
            if not args.no_llm and not args.raw_only:
                OllamaClient().ready()
            output = new_output(args.output)
            token = None if args.offline or args.raw_only else get_token()
            result = process(args.audio, token=token, offline=args.offline, alignment=args.alignment,
                             output_dir=output, raw_only=args.raw_only)
            if args.raw_only:
                print((output / "01_raw.txt").read_text(encoding="utf-8"))
                print(f"Raw transcript saved to {output}")
                return 0
        if interactive or args.rename:
            args.name_a = input("Name for Speaker A [Speaker A]: ").strip() or args.name_a
            args.name_b = input("Name for Speaker B [Speaker B]: ").strip() or args.name_b
        result["speakers"].update(A=args.name_a, B=args.name_b)
        if args.rename:
            output = new_output(args.output)
        save_speakers(result, output)
        export(result, output)  # Legacy stage-2 filenames remain available.
        if args.rename or args.no_llm:
            print(render(result))
        else:
            edited = enhance(result)
            save_readable(edited, output)
            print(readable_text(edited))
            print("For the separate comparison pass, run --review on 03_readable.json.")
        print(f"Completed stages saved to {output}")
        return 0
    except KeyboardInterrupt:
        print("\nProcessing cancelled. Completed stage files remain on disk.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}\nCompleted earlier stages remain on disk, if any.", file=sys.stderr)
        return 1
