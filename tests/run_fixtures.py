"""Readable (03) transcripts built through the real merge/enhance code with a scripted local LLM."""
from interview_local.core import Turn, Word, merge
from interview_local.llm import enhance


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
    return [Word(start, end, text, probability) for (start, end), text, probability in zip(times, texts, probabilities)]


INTERVIEW_TURNS = [Turn(0, 4, "first"), Turn(4.5, 9, "second")]


def readable_transcript(paragraphs, words=None, regular_turns=None):
    """Return a readable transcript of interview_words edited into the given paragraphs.

    Args:
        paragraphs: (first_word, last_word, speaker, text) tuples the scripted LLM returns.
        words: Source words; interview_words() by default.
        regular_turns: Regular diarization turns for overlap detection.
    """
    source = merge(words or interview_words(), INTERVIEW_TURNS, regular_turns)
    source["speakers"].update(A="Anna", B="Bernd")
    return enhance(source, client=ScriptedClient(paragraphs), status=lambda _: None)


TWO_PARAGRAPHS = [(0, 3, "A", "Guten Tag, wie geht's?"), (4, 7, "B", "Danke, gut. Und selbst?")]
