# Interview Local

A local Python app for **German interview transcription**, **two-speaker attribution with overlap detection**, and a **readable Ministral draft**. A separate comparison pass can flag uncertain edits. Originals remain available throughout the workflow.

Code, interface and documentation are English. Speech content stays in German. Primary platform: macOS on Apple Silicon. Windows/Linux launch paths are provided but not verified here; Intel Macs are not supported by the pinned PyTorch 2.8 environment.

## Install and start

1. Extract the app ZIP into a writable folder you intend to keep.
2. Install Python 3 from [python.org](https://www.python.org/downloads/) if necessary. Include the Python launcher on Windows.
3. Install and open [Ollama](https://ollama.com/download) for the readable LLM stage.
4. macOS: open **`Install.command`**, or run `bash Install.command` in Terminal. Windows: open **`Install.bat`**.
5. On macOS the installer prepares Ministral automatically if Ollama is installed and running. Otherwise run `Start.command --prepare-llm` (Windows: `Start.bat --prepare-llm`) after starting Ollama.
6. Open **`Start.command`** / **`Start.bat`**, enter a path or drag a WAV, MP3 or M4A file into the terminal, then enter speaker names when prompted.

Python dependencies are isolated in `.venv` with Python 3.12 and `uv.lock`. `.installer` contains uv. The initial setup requires internet access and several GB of downloads. Plan for roughly **25 GB free disk space** including model caches and temporary files. Nothing is uploaded to an inference service.

### Do I need a Hugging Face account?

**Not for Ministral or Whisper.** Ministral downloads from the Ollama registry without an HF login; the public faster-whisper model is downloaded anonymously.

The ready-to-use ZIP includes the already downloaded **community-1 model in `models/community-1/`**, with its source/revision, attribution and CC-BY-4.0 license link in `NOTICE.md`. With that complete folder, pyannote loads directly from disk and the app does not request an HF token. The optional model weights are separate from the MIT app license. Retain their notices when sharing them.

The smaller **source-only ZIP and a Git checkout exclude model weights**. For those installations, either place a complete community-1 folder at `models/community-1/`, set `INTERVIEW_DIARIZATION_MODEL` to such a folder, or use the official HF download. The latter requires an HF account, acceptance of the [community-1 conditions](https://huggingface.co/pyannote/speaker-diarization-community-1), and an authorized [read token](https://huggingface.co/settings/tokens). The app keeps entered tokens only for the run; existing HF login and `HF_TOKEN` are also supported. Do not put tokens in source, command arguments or version control.

After initial downloads, models can run without internet. `--offline` prevents HF/Transformers downloads and the app uses only the local Ollama service for editing. Optional alignment also needs its language data prepared beforehand. No cloud fallback is implemented. HF/pyannote/ONNX Runtime telemetry is disabled.

## Three preserved stages, plus separate review

Every run gets a new directory under `transcripts/`. Existing nonempty output directories are rejected. Completed stages remain saved if a later stage fails or is cancelled.

| Files | Purpose |
|---|---|
| `01_raw.txt`, `01_raw.json` | Original Whisper output, without speaker attribution or LLM changes. Saved immediately after speech recognition. |
| `02_speakers.txt`, `02_speakers.json` | Original words plus speaker IDs, timestamps, uncertainty and simultaneous-speech markers. Includes the raw transcript for provenance. |
| `03_readable.txt`, `03_readable.md`, `03_readable.json` | Human-readable LLM draft with speaker paragraphs and time ranges. JSON is an audit artifact, not the main reading format. |
| `04_review.txt`, `04_review.md`, `04_review.json` | Separately requested consistency comparison against the embedded source, with paragraph uncertainty labels and reasons. |
| `source.wav` / `source.mp3` / `source.m4a` | Web UI runs only: the stored copy of the uploaded recording, used for playback in the editor. |
| `corrections.json` | Web UI runs only: your manual corrections (paragraph text, speaker assignment, speaker names) and which paragraphs you have checked. The stage files above stay unchanged. Files saved before the checked status existed open with no paragraph checked. |

Every word in `01_raw.json` and `02_speakers.json` stores Whisper's word `probability` (0–1). Words without a corresponding Whisper word — whole-segment fallback text and words from the optional `--align` path — store `probability: null`.

`transcript.txt` and `transcript.json` remain aliases of the speaker-attributed stage for compatibility. The raw and speaker versions are not overwritten by the LLM or reviewer.

Open **`03_readable.md` or `03_readable.txt`** to read the interview. The app prints this readable draft to the terminal too. Review is deliberately a separate command rather than silently applied to every run.

```bash
# Prepare speech models; a local community-1 folder avoids HF authentication
.venv/bin/python -m interview_local --prepare
# Prepare the local LLM (Ollama must be running)
.venv/bin/python -m interview_local --prepare-llm
# Optional alignment model and NLTK data
.venv/bin/python -m interview_local --prepare --align

# Full three-stage workflow using prepared models
.venv/bin/python -m interview_local --offline "/path/to/interview.m4a" \
  --name-a "Interviewer" --name-b "Participant" --output transcripts/interview-01

# Raw speech recognition only: no HF account, no diarization and no LLM
.venv/bin/python -m interview_local --raw-only "/path/to/interview.wav"
# Stop after speaker attribution
.venv/bin/python -m interview_local --offline --no-llm "/path/to/interview.wav"

# Run the LLM later, without rerunning the speech models
.venv/bin/python -m interview_local --enhance transcripts/interview-01/02_speakers.json \
  --output transcripts/interview-01-readable

# Separate comparison pass: readable draft versus its embedded original input
.venv/bin/python -m interview_local --review transcripts/interview-01-readable/03_readable.json \
  --output transcripts/interview-01-reviewed

# Rename a speaker transcript without inference (old JSON files are supported too)
.venv/bin/python -m interview_local --rename transcripts/interview-01/02_speakers.json \
  --output transcripts/interview-01-renamed
.venv/bin/python -m interview_local --doctor
```

The development launchers reuse `.cache/models` and `.cache/nltk_data` when present. For direct Python commands on this computer, select these caches explicitly:

```bash
HF_HUB_CACHE="$PWD/.cache/models" NLTK_DATA="$PWD/.cache/nltk_data" \
  .venv/bin/python -m interview_local --offline "/path/to/interview.wav"
```

The prepared community-1 folder is discovered independently of HF cache settings. On Windows, replace `.venv/bin/python` with `.venv\Scripts\python.exe`.

## What the LLM may change

The local model is **Ministral 3 8B Instruct**, Ollama tag `ministral-3:8b`, not a cloud model. It receives bounded chunks of at most 120 words, neighboring context, speaker IDs and uncertainty/overlap information. The model digest and quantization are recorded in the output. The endpoint is restricted to loopback HTTP; redirects, environment proxies and cloud-backed model entries are rejected.

The editor improves punctuation, capitalization and paragraph grouping. It is instructed to retain dialect, repetitions, fillers, names, numbers, negations and interruptions. Small wording or speaker corrections are proposals and always trigger uncertainty. A broad rewrite is rejected. Every original word range must be covered exactly once and in order. The application computes timestamps from those word ranges; the model cannot invent new times.

Invalid responses are retried once. If validation still fails, the chunk retains its original wording and is explicitly flagged. Nearby paragraphs from the same speaker are grouped into readable blocks. The audit JSON retains original text, source word IDs, editorial reasons, wording/speaker-change flags and a source fingerprint.

The separate review call compares edited paragraphs with their source words. It does not overwrite the edited draft. Existing source, overlap and edit-change flags cannot be erased by the reviewer. This is a **text consistency check, not audio verification**; a fluent sentence or agreement between two calls to the same model does not establish correctness. New issues are labeled uncertain rather than silently treated as resolved.

## Speaker overlap and uncertainty

JSON stores both `diarization` (exclusive turns used for assignment) and `regular_diarization` (which may contain simultaneous voices). `overlaps` lists intervals with multiple active speaker IDs. Words/segments intersecting such an interval receive overlap markers and uncertainty reasons. A legacy JSON without regular turns is marked `overlap_detection_available: false`, rather than pretending that overlap was checked.

The `uncertain` Boolean is a **heuristic, not a calibrated probability**. It covers missing/ambiguous speaker matches, zero/invalid word duration, fallback timing, simultaneous voices and subsequent LLM changes. `speaker: null` means unassigned. A false uncertainty flag does not guarantee accuracy. Text alone cannot reliably decide which voice uttered an overlapping word or recover speech that Whisper missed.

Speaker A is the first detected voice; labels are not real-person identification. Exactly two voices is a model constraint, not evidence that the recording really contains two speakers.

## Hardware and processing

Development hardware: Apple M3 Pro, 12 CPU cores, 18 GPU cores, **18 GB unified RAM**, macOS 26.6.2. The requested 16-GB target remains conservative and long recordings on actual 16-GB hardware have not been measured.

- Whisper large-v3: faster-whisper/CTranslate2, CPU INT8, beam 5, at most six threads, one worker and no transcription batches. Metal is not used by this backend.
- pyannote/optional WhisperX alignment: CPU Float32, at most four threads, segmentation/embedding batches 1.
- Speech models run sequentially in separate processes, which exit before the LLM starts. Decoding is streamed; pyannote receives the complete original waveform and alignment reads one ASR segment at a time. No speech regions are trimmed before either main model.
- Ministral runs locally in Ollama with an 8,192-token context and one sequential request at a time. It can use Apple Metal. The app asks Ollama to unload this model after editing/review; it does not stop other user models.
- Historical two-minute speech test peaks were 3.20 GiB process RSS for Whisper, 0.96 GiB for alignment and 0.81 GiB for pyannote. These are not total system memory. Apple CPU/GPU share the same RAM; there is no additional dedicated VRAM. Model weight sizes are not measured peak RAM.

Alignment stays opt-in (`--align`): on the original telephone test it worsened speaker-attributed word error from 37.57% to 44.97%. It changes word timing, not recognized text. There is no verified maximum recording duration. Close memory-intensive applications when needed.

## Local web UI and transcript editor

```bash
.venv/bin/python -m interview_local --web
# Keep web runs elsewhere (default: transcripts/)
.venv/bin/python -m interview_local --web --runs-dir /path/to/runs
```

The server listens on `127.0.0.1` with a random free port, prints its URL and opens it in the browser. It is not reachable from other machines and rejects requests that do not address `127.0.0.1:<port>` or `localhost:<port>`. Press Ctrl+C in the terminal to stop it.

- **New transcription:** choose a WAV, MP3 or M4A file and enter both speaker names. The recording is copied into a new run directory as `source.<ext>`. The page shows the running step (speech recognition, speaker recognition, LLM) and any error. One transcription runs at a time and cannot be cancelled from the browser. Ollama must be running with Ministral prepared. Speaker recognition needs the local `models/community-1` folder, `HF_TOKEN` or a cached `hf auth login`; the web UI never asks for a token. The web flow does not use `--align`.
- **Editor:** paragraphs with speaker name and time range. Blue words differ from the speech recognition output (hover for the original word, or tick “Show original words”); words with a wavy red underline have a Whisper word probability below 0.5 or none at all. Words with a dashed purple line above them follow at least 5 seconds of silence and are possibly invented text: Whisper tends to hallucinate in long silences. Orange paragraphs need a speaker check: simultaneous speech, unclear speaker boundary, no speaker match, unassigned, or speaker changed by the LLM. Assigning a paragraph manually clears this highlight.
- **Corrections:** “Edit” turns a paragraph into a text field. Select words and use “Assign selection”, or “Assign paragraph”, to give text to a speaker; the paragraph is split accordingly. “Swap names” exchanges the two speaker names. Corrections are saved immediately to `corrections.json`; reopening the run from the start page shows them again.
- **Review status:** tick “checked” in a paragraph's header once you have reviewed it; the toolbar shows “x of n paragraphs checked”. Editing a paragraph's text, “Assign paragraph” and “Assign selection” mark the resulting paragraphs as checked; untick to undo. The status is saved in `corrections.json`.
- **Audio:** click a word to play the recording from that word. Resuming after a pause starts 2 seconds earlier. “Speed” sets playback to 0.75×, 1×, 1.25× or 1.5×. The word being spoken is outlined, and the editor scrolls to follow it unless a paragraph is open for editing.
- **Export:** “Export .md” / “Export .txt” download the corrected transcript, without review marks or checked status.

Only runs started from the web UI can be opened in the editor: older runs have no stored audio copy and no word probabilities. If `03_readable.json` of a run changes after corrections were saved, the editor refuses to open it rather than misapplying them.

## Optional native window

```bash
.venv/bin/python -m interview_local --gui
```

The window provides audio selection, names for a new run, model preparation, a separate review action and Raw/Speakers/Readable/Review tabs. Stages are saved automatically into a newly created run directory chosen at start. Tk is optional; without it the app falls back to the terminal. The window logic is tested against an in-memory Tk stand-in (`tests/support/fake_tk.py`); real Tk widgets and the Tk setup are not exercised by the tests. Python 3.12 from python.org can be used to create a Tk-capable environment with `uv sync --frozen --python /full/path/to/python3.12`.

## Troubleshooting

- **Cannot reach local Ollama:** Install/open Ollama, then run `--prepare-llm`. The raw/speaker workflow remains available with `--no-llm`.
- **Ministral missing:** Run `--prepare-llm` once with internet. No HF token is involved.
- **401/403:** This concerns an HF model download, not Ministral. Provide the local community-1 folder or accept its HF conditions and renew your read token/login (`.venv/bin/hf auth login`).
- **Missing offline models:** Prepare them with internet before using `--offline`.
- **LLM response rejected:** Original wording is retained and flagged. Inspect the audit JSON; no invented or missing source ranges are accepted.
- **Later step failed:** Keep `01_raw.*` and `02_speakers.*`. Run `--enhance` later; do not repeat expensive speech processing unnecessarily.
- **Low memory:** Close other apps. The app releases its speech-model processes before invoking the LLM. It does not evict unrelated Ollama models.
- **TorchCodec messages:** Its optional system decoder is unused. PyAV/SoundFile decode audio and pyannote receives a waveform directly. See TEST_REPORT.md.
- **Invalid audio:** Check it in an audio player and try WAV. DRM-protected files are unsupported.
- **Launcher blocked:** Run `bash Install.command` / `bash Start.command` in the extracted folder.

## Architecture, tests and GitHub

`core.py` contains model-independent assignment, overlap detection and export. `backends.py` adapts Whisper/pyannote/WhisperX; `worker.py` isolates speech models; `pipeline.py` persists completed stages. `llm.py` implements bounded local editing, source validation and comparison review. `artifacts.py` writes human-readable and audit outputs. Other speech models such as Parakeet/Nemotron would need their own adapters in `backends.py` and worker stages that save the same JSON results. LLM behavior can be replaced behind the client interface.

```bash
uv sync --frozen --python 3.12
uv run --frozen python -m unittest discover -s tests -v
uv run --frozen interview-local --help
```

The public [CALLHOME German sample](https://catalog.ldc.upenn.edu/LDC2026S06) is downloadable separately with `tools/fetch_test_sample.py test_data/callhome`. It is not bundled in either app ZIP. Convert its FLAC to WAV with a separately installed FFmpeg if necessary. The scorer uses `--offset 119 --start 0.71 --end 103.22` for this sample only. Reference normalization and historical results are in [TEST_REPORT.md](TEST_REPORT.md) and `benchmark/comparison.json`.

The source lives in the private repository [leonard-kunz-taod/interview-local](https://github.com/leonard-kunz-taod/interview-local). `.gitignore` excludes `models/`, caches, environments, token files, audio and default transcript outputs. The source-only ZIP contains no model weights.

## Sources and licenses

App code: MIT (LICENSE). Dependencies, weights and recordings retain their licenses.

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper): local CPU/INT8 transcription.
- [community-1](https://huggingface.co/pyannote/speaker-diarization-community-1): CC-BY-4.0; source/revision/attribution are in the prepared folder's NOTICE.md. [License](https://creativecommons.org/licenses/by/4.0/).
- [Ministral 3 8B Instruct](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512): Apache 2.0; downloaded separately through [Ollama](https://ollama.com/library/ministral-3:8b).
- [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs): schema-constrained local responses, still requiring application validation.
- [WhisperX](https://github.com/m-bain/whisperX) and the optional [German VoxPopuli alignment model](https://huggingface.co/facebook/wav2vec2-base-10k-voxpopuli-ft-de). The latter is CC-BY-NC-4.0; the default without alignment does not use it. Commercial alignment requires a suitable model/license.
