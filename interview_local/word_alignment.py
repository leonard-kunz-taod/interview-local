"""Map a paragraph's display text back to the source (ASR) words it came from."""
from __future__ import annotations
from dataclasses import dataclass
import difflib
from typing import Literal

from .llm import tokens

Change = Literal["unchanged", "replaced", "inserted"]


@dataclass(frozen=True)
class DisplayWord:
    """One whitespace-separated word of a paragraph text, linked to its source words.

    Attributes:
        text: The displayed word, including attached punctuation.
        source_word_ids: Global ids of the source words it covers; empty when inserted.
        change: Whether the word differs from its source words.
        original_text: The covered source words joined by spaces; empty when inserted.
        removed_before: Source words dropped from the text at this position.
    """
    text: str
    source_word_ids: tuple[int, ...]
    change: Change
    original_text: str
    removed_before: str


def align_text_to_source(text: str, source_words: list[dict]) -> list[DisplayWord]:
    """Align display text with source words on casefolded word tokens.

    Case and punctuation differences are not changes, matching
    ``content_changed`` in ``llm.validate_edit``.

    Args:
        text: The paragraph text shown to the user.
        source_words: Source word dicts with ``word_id`` and ``text``, in order.

    Returns:
        One DisplayWord per whitespace-separated word with word characters.

    Raises:
        ValueError: If the text contains no words or the source is empty.
    """
    if not source_words:
        raise ValueError("Cannot align text to an empty source.")
    display_texts = _display_texts(text)
    text_tokens = [(token, index) for index, word in enumerate(display_texts) for token in tokens(word)]
    source_tokens = [(token, word["word_id"]) for word in source_words for token in tokens(word["text"])]
    source_text_by_id = {word["word_id"]: word["text"] for word in source_words}
    alignment = _TokenAlignment(len(display_texts))
    matcher = difflib.SequenceMatcher(a=[t for t, _ in text_tokens], b=[t for t, _ in source_tokens],
                                      autojunk=False)
    for operation, text_begin, text_end, source_begin, source_end in matcher.get_opcodes():
        display_indexes = [index for _, index in text_tokens[text_begin:text_end]]
        source_ids = [word_id for _, word_id in source_tokens[source_begin:source_end]]
        if operation == "equal":
            for index, word_id in zip(display_indexes, source_ids):
                alignment.link(index, [word_id], changed=False)
        elif operation == "replace":
            alignment.link_block(display_indexes, source_ids)
        # Opcodes turn the text into the source: "delete" is text-only, "insert" is source-only.
        elif operation == "delete":
            for index in display_indexes:
                alignment.link(index, [], changed=True)
        elif operation == "insert":
            next_index = text_tokens[text_begin][1] if text_begin < len(text_tokens) else len(display_texts) - 1
            alignment.delete(next_index, source_ids)
    return alignment.display_words(display_texts, source_text_by_id)


def _display_texts(text: str) -> list[str]:
    """Split text on whitespace and attach punctuation-only pieces to a neighbouring word."""
    pieces = text.split()
    words: list[str] = []
    leading_punctuation: list[str] = []
    for piece in pieces:
        if tokens(piece):
            words.append(" ".join(leading_punctuation + [piece]))
            leading_punctuation = []
        elif words:
            words[-1] += " " + piece
        else:
            leading_punctuation.append(piece)
    if not words:
        raise ValueError("The text contains no words.")
    return words


class _TokenAlignment:
    """Accumulates per-display-word links while walking the token opcodes."""

    def __init__(self, display_count: int):
        self.source_ids: list[list[int]] = [[] for _ in range(display_count)]
        self.changed = [False] * display_count
        self.deleted_ids: list[tuple[int, int]] = []

    def link(self, index: int, source_ids: list[int], changed: bool) -> None:
        for word_id in source_ids:
            if word_id not in self.source_ids[index]:
                self.source_ids[index].append(word_id)
        self.changed[index] |= changed

    def link_block(self, display_indexes: list[int], source_ids: list[int]) -> None:
        """Link a replaced block: pair token by token when lengths match, else share all ids."""
        if len(display_indexes) == len(source_ids):
            for index, word_id in zip(display_indexes, source_ids):
                self.link(index, [word_id], changed=True)
            return
        for index in display_indexes:
            self.link(index, source_ids, changed=True)

    def delete(self, next_index: int, source_ids: list[int]) -> None:
        for word_id in source_ids:
            self.deleted_ids.append((next_index, word_id))

    def display_words(self, display_texts: list[str], source_text_by_id: dict[int, str]) -> list[DisplayWord]:
        removed_before: list[list[int]] = [[] for _ in display_texts]
        for next_index, word_id in self.deleted_ids:
            owner = self._owner_of(word_id)
            if owner is not None:
                self.changed[owner] = True
            elif word_id not in removed_before[next_index]:
                removed_before[next_index].append(word_id)
        result = []
        for index, text in enumerate(display_texts):
            ids = tuple(sorted(self.source_ids[index]))
            if not self.changed[index]:
                change: Change = "unchanged"
            elif ids:
                change = "replaced"
            else:
                change = "inserted"
            result.append(DisplayWord(
                text=text, source_word_ids=ids, change=change,
                original_text=" ".join(source_text_by_id[i] for i in ids),
                removed_before=" ".join(source_text_by_id[i] for i in removed_before[index])))
        return result

    def _owner_of(self, word_id: int) -> int | None:
        """Return the last display word linked to a partially deleted source word, if any."""
        owners = [index for index, ids in enumerate(self.source_ids) if word_id in ids]
        return owners[-1] if owners else None
