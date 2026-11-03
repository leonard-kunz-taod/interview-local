# Test report — November 3, 2026

The complete terminal workflow **audio → transcription → diarization → word assignment → TXT/JSON** passed an offline test. A separate complete offline run also tested optional German word alignment. This is a working first version requiring manual review, not a promise of error-free interview analysis.

## Recording and reference

Source: the official public [CALLHOME German Second Edition sample, LDC2026S06](https://catalog.ldc.upenn.edu/LDC2026S06). It is a real 120-second German telephone conversation at 8 kHz with two recorded channels, converted to WAV for the app. Both models received the same complete mono downmix; the separate channels were not used to bypass diarization.

The supplied TSV contains 49 utterance intervals with text and speaker labels A/B. It uses the original recording's timeline, while the sample starts at original second 119. Several uniquely matched utterance starts supported this offset (median observed difference 118.975 seconds).

The evaluated window was **0.71–103.22 seconds** of the sample, including 250 ms tolerance around the supplied reference region. The remaining audio has no supplied reference and was excluded. There were **338 normalized reference words** and about **24.66 seconds of overlapping voices**. Dialect, narrow-band telephone audio and short interruptions make this a difficult sample.

The provider's reference was not independently reannotated word by word. One sample cannot establish general interview quality, and this is not an official CALLHOME benchmark.

## Quality comparison

| Metric | Default: Whisper word timestamps | Optional: WhisperX word alignment |
|---|---:|---:|
| Word error rate (WER) | **25.44%** | 26.04% |
| Word errors / reference words | 86 / 338 | 88 / 338 |
| Speaker-attributed word error rate (cpWER) | **37.57%** | 44.97% |
| Diarization error rate (DER) | 24.29% | 24.29% |
| Detected voices | 2 | 2 |

**Default decision:** Word alignment works technically but worsened combined word/speaker error on this recording. It is therefore opt-in with `--align`. Alignment does not change recognized text; the small WER difference comes from selecting words at the timed evaluation boundaries.

WER lowercases text and removes punctuation/noise labels, without expanding numbers. cpWER compares speaker transcripts under the best global A/B permutation and counts unassigned words as additional errors. The chronological order of overlapping voices can affect ordinary WER.

DER uses pyannote.metrics with a 250-ms collar, overlap included and utterance-level reference intervals. Components were **21.17 s missed speech, 2.70 s speaker confusion and 0.25 s false alarm**, relative to 99.28 s evaluated reference-speaker time after applying the collar. DER is not the proportion of incorrectly assigned words. The exclusive output cannot represent both simultaneous voices; this is a substantial limitation.

Machine-readable measurements, source hashes and model revisions are in [benchmark/comparison.json](benchmark/comparison.json). Evaluation commands are in README.md.

## Runtime and memory

Hardware: MacBook Pro, Apple M3 Pro, 12 CPU cores, **18 GB unified RAM**, macOS 26.6.2; Python 3.12.11. CPU inference, Whisper INT8, pyannote/alignment Float32, with models in separate sequential processes.

| Stage | Default run time | Peak process RSS |
|---|---:|---:|
| Access check | 0.2 s | 0.04 GiB |
| WAV decoding | 0.3 s | 0.05 GiB |
| Whisper large-v3 | 125.2 s | 3.20 GiB |
| community-1 | 44.4 s | 0.81 GiB |
| Optional alignment, separate comparison run | 13.0 s | 0.96 GiB |

The default run took approximately **170 seconds excluding downloads**. Worker elapsed times and maximum RSS were measured with `resource.getrusage`. RSS does not include the rest of the system and is not dedicated GPU memory. The earlier comparison run took 143.6 s for Whisper at 3.02 GiB RSS; timings vary. Long interviews and an actual 16-GB computer remain untested.

## Checks completed

- Isolated Python 3.12 installation using uv and pinned package versions; dependency validation passed. The original ZIP was separately extracted and installed with `uv sync --frozen --offline`; the installed CLI and dependencies were checked.
- Real imports of faster-whisper, pyannote.audio and WhisperX.
- WAV, MP3 and M4A decoding, including Whisper decoding after PCM conversion. Format probes were short tones; complete model runs used the WAV telephone sample. The one-second M4A probe included approximately 22 ms codec padding.
- Two complete offline CLI runs, with and without alignment, producing TXT and structured JSON.
- All 359 words in the alignment run were retained and aligned; seven remained unassigned and eight were flagged uncertain.
- CLI speaker renaming, UTF-8 exports and protection against overwriting existing output files.
- Seven automated tests covering speaker changes/gaps, ambiguity/zero durations, renaming/export, scoring normalization, sequential orchestration/error cleanup and preservation of unalignable words.
- Public downloads of Whisper and alignment weights. community-1 inference used existing local weights. A fresh gated configuration download also passed after renewing the HF login.

## Integration fixes

- faster-whisper 1.2.1 fails with PyAV 19's removed decoding argument. PyAV is constrained to `<19` and decoding was retested.
- TorchCodec is pinned to 0.7 for PyTorch 2.8. Its optional system decoder still could not load here. The tested PyAV/SoundFile path passes waveforms directly to pyannote, so that decoder is unused and its known import warning is selectively suppressed.
- HF downloads use direct HTTP by default after the Xet transfer showed no progress in the test.

## English codebase update (0.1.1)

The package/CLI were renamed to `interview_local`/`interview-local`; launchers are now `Install.command`, `Start.command`, `Install.bat` and `Start.bat`. New exports use `transcript.txt` and `transcript.json`. Source comments, interface text, documentation and benchmark descriptions are English. German speech fixtures intentionally remain German to exercise Unicode handling. Existing recorded benchmark numbers and transcription content were not changed.

This update changes names and interface text, not the inference/assignment algorithms. The existing model measurements above are historical results from 0.1.0. Regression tests, installed CLI checks, renamed-worker checks and archive validation are repeated for 0.1.1 without rerunning expensive inference solely for a language change.

## Still untested

- The optional Tk window: `_tkinter` was unavailable. The terminal path was tested.
- Windows, Linux, Windows batch execution and macOS Finder double-click execution. macOS shell scripts were syntax checked; uv installation was actually exercised.
- A complete community-1 weight download using a newly created account and first-time access-consent flow; existing weights passed offline inference.
- Long recordings, other microphones/noise conditions, other dialects, actual 16-GB systems and automatic recovery after a hard crash.
- Whether optional alignment improves other recordings. Its negative result on this sample is retained explicitly.

## Uncertainty semantics

The JSON `uncertain` field is a **heuristic Boolean, not a calibrated probability**. It flags missing/ambiguous temporal overlap, invalid word durations and fallback/estimated alignment. `false` does not certify correctness. Overlapping voices and short interruptions can be assigned incorrectly without triggering the flag.
