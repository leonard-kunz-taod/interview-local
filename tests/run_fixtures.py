"""Readable (03) transcripts built through the real merge/enhance code with a scripted local LLM."""
from collections.abc import Sequence
import json

from interview_local.core import Turn, Word, merge
from interview_local.corrections import readable_source_words, reassign_words
from interview_local.llm import enhance

ASR_OUTPUT = {"segments": [dict(start=0, end=1, text="Guten Tag", words=[
    dict(start=0, end=.5, text="Guten", timing="whisper", probability=.9),
    dict(start=.5, end=1, text="Tag", timing="whisper", probability=.4)])], "language": "de"}
DIARIZATION_OUTPUT = {"exclusive": [dict(start=0, end=1, speaker="s0")],
                      "regular": [dict(start=0, end=1, speaker="s0")]}
STAGE_OUTPUTS = {"decode": "", "asr": json.dumps(ASR_OUTPUT), "diarize": json.dumps(DIARIZATION_OUTPUT)}


def speech_models(failing_stage=None):
    """Stand-in for the model worker processes: writes the outputs the real stages produce."""
    def worker(stage, source, target, *args):
        if stage == failing_stage:
            raise RuntimeError(f"Stage '{stage}' failed: model unavailable")
        if stage in STAGE_OUTPUTS:
            target.write_text(STAGE_OUTPUTS[stage])
    return worker


class ScriptedClient:
    """Stands in for local Ollama: returns fixed paragraphs for the single edit chunk."""

    def __init__(self, paragraphs):
        self.paragraphs = paragraphs

    def ready(self):
        return {"model": "scripted-test"}

    def unload(self):
        pass

    def chat(self, instruction, data, schema):
        return {"paragraphs": [dict(first_word=first, last_word=last, speaker=speaker, text=text,
                                    uncertain=False, reason="test") for first, last, speaker, text in self.paragraphs]}


def interview_words(probabilities=None):
    """Eight German words, A speaks 0-3 (0-4 s), B speaks 4-7 (5-9 s)."""
    texts = ["Guten", "Tag", "wie", "geht's", "Danke", "gut", "und", "selbst"]
    times = [(0, .5), (.5, 1), (1, 1.5), (1.5, 2), (5, 5.5), (5.5, 6), (6, 6.5), (6.5, 7)]
    probabilities = probabilities or [.9] * len(texts)
    return [Word(start, end, text, probability, "whisper") for (start, end), text, probability in zip(times, texts, probabilities)]


INTERVIEW_TURNS = [Turn(0, 4, "first"), Turn(4.5, 9, "second")]


def readable_transcript(paragraphs, words=None, regular_turns: Sequence[Turn] = ()):
    """Return a readable transcript of interview_words edited into the given paragraphs.

    Args:
        paragraphs: (first_word, last_word, speaker, text) tuples the scripted LLM returns.
        words: Source words; interview_words() by default.
        regular_turns: Regular diarization turns for overlap detection.
    """
    source = merge(words or interview_words(), INTERVIEW_TURNS, list(regular_turns))
    source["speakers"].update(A="Anna", B="Bernd")
    return enhance(source, client=ScriptedClient(paragraphs), status=lambda _: None)


TWO_PARAGRAPHS = [(0, 3, "A", "Guten Tag, wie geht's?"), (4, 7, "B", "Danke, gut. Und selbst?")]


def reassign(readable, paragraphs, first_word, last_word, speaker):
    """Reassign a word range of a readable transcript's corrected paragraphs to a speaker."""
    return reassign_words(paragraphs, readable_source_words(readable), set(readable["speakers"]),
                          first_word, last_word, speaker)
