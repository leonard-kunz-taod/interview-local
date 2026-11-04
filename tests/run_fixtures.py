"""Readable (03) transcripts built through the real merge/enhance code with a scripted local LLM."""

import json
from collections.abc import Sequence

from support.fake_ollama import FakeOllama, ScriptedModel, served_by

from interview_local.core import Turn, Word, merge
from interview_local.corrections import readable_source_words, reassign_words
from interview_local.llm import enhance

ASR_OUTPUT = {
    "segments": [
        {
            "start": 0,
            "end": 1,
            "text": "Guten Tag",
            "words": [
                {
                    "start": 0,
                    "end": 0.5,
                    "text": "Guten",
                    "timing": "whisper",
                    "probability": 0.9,
                },
                {
                    "start": 0.5,
                    "end": 1,
                    "text": "Tag",
                    "timing": "whisper",
                    "probability": 0.4,
                },
            ],
        }
    ],
    "language": "de",
}
DIARIZATION_OUTPUT = {
    "exclusive": [{"start": 0, "end": 1, "speaker": "s0"}],
    "regular": [{"start": 0, "end": 1, "speaker": "s0"}],
}
ALIGNMENT_OUTPUT = [
    {"start": 0, "end": 0.5, "text": "Guten", "timing": "aligned", "probability": None},
    {"start": 0.5, "end": 1, "text": "Tag", "timing": "aligned", "probability": None},
]
STAGE_OUTPUTS = {
    "decode": "",
    "asr": json.dumps(ASR_OUTPUT),
    "align": json.dumps(ALIGNMENT_OUTPUT),
    "diarize": json.dumps(DIARIZATION_OUTPUT),
}
AUDIO_READING_STAGES = {"decode", "asr", "align", "diarize"}
GATED_STAGES = {"diarize", "prepare-diarize"}


def speech_models(failing_stage=None):
    """Stand-in for the model worker processes: like the real stages, each reads its input file, reports
    progress and writes the output the real stage produces."""

    def worker(stage, source, target, status, token, offline=False):
        if stage == failing_stage:
            raise RuntimeError(f"Stage '{stage}' failed: model unavailable")
        if stage in AUDIO_READING_STAGES and not source.is_file():
            raise RuntimeError(f"Stage '{stage}' failed: {source} does not exist")
        status(f"Loading {stage} model")
        if stage in STAGE_OUTPUTS:
            target.write_text(STAGE_OUTPUTS[stage])

    return worker


def gated_speech_models(access_token):
    """Speech-model stand-in whose speaker recognition, like the gated pyannote model, needs the HF token."""
    models = speech_models()

    def worker(stage, source, target, status, token, offline=False):
        if stage in GATED_STAGES and token != access_token:
            raise RuntimeError(f"Stage '{stage}' failed: 401 Unauthorized")
        models(stage, source, target, status, token, offline)

    return worker


def uncached_speech_models():
    """Speech-model stand-in with an empty model cache: every stage fails when downloads are forbidden."""
    models = speech_models()

    def worker(stage, source, target, status, token, offline=False):
        if offline:
            raise RuntimeError(f"Stage '{stage}' failed: model not in the local cache")
        models(stage, source, target, status, token, offline)

    return worker


def scripted_ollama(paragraphs):
    """Local Ollama whose model answers the single edit chunk with fixed (first_word, last_word, speaker, text)
    paragraphs."""
    reply = {
        "paragraphs": [
            {
                "first_word": first,
                "last_word": last,
                "speaker": speaker,
                "text": text,
                "uncertain": False,
                "reason": "test",
            }
            for first, last, speaker, text in paragraphs
        ]
    }
    return FakeOllama(ScriptedModel(response=reply))


def interview_words(probabilities=None):
    """Eight German words, A speaks 0-3 (0-4 s), B speaks 4-7 (5-9 s)."""
    texts = ["Guten", "Tag", "wie", "geht's", "Danke", "gut", "und", "selbst"]
    times = [
        (0, 0.5),
        (0.5, 1),
        (1, 1.5),
        (1.5, 2),
        (5, 5.5),
        (5.5, 6),
        (6, 6.5),
        (6.5, 7),
    ]
    probabilities = probabilities or [0.9] * len(texts)
    return [
        Word(start, end, text, probability, "whisper")
        for (start, end), text, probability in zip(times, texts, probabilities)
    ]


INTERVIEW_TURNS = [Turn(0, 4, "first"), Turn(4.5, 9, "second")]


def readable_transcript(paragraphs, words=None, regular_turns: Sequence[Turn] = ()):
    """A readable transcript of the source words edited into ``paragraphs``: (first_word, last_word, speaker, text) tuples."""
    source = merge(words or interview_words(), INTERVIEW_TURNS, list(regular_turns))
    source["speakers"].update(A="Anna", B="Bernd")
    with served_by(scripted_ollama(paragraphs)):
        return enhance(source, lambda _: None)


TWO_PARAGRAPHS = [
    (0, 3, "A", "Guten Tag, wie geht's?"),
    (4, 7, "B", "Danke, gut. Und selbst?"),
]


def reassign(readable, paragraphs, first_word, last_word, speaker):
    """Reassign a word range of a readable transcript's corrected paragraphs to a speaker."""
    return reassign_words(
        paragraphs,
        readable_source_words(readable),
        set(readable["speakers"]),
        first_word,
        last_word,
        speaker,
    )
