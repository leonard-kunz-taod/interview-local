"""Map a paragraph's display text back to the source (ASR) words it came from."""
from __future__ import annotations
from dataclasses import dataclass
import difflib
from itertools import pairwise
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
    alignment = _TokenAlignment(len(display_texts))
    for opcode in _token_opcodes(_display_tokens(display_texts), _source_tokens(source_words)):
        alignment.apply(opcode)
    return alignment.display_words(display_texts, {word["word_id"]: word["text"] for word in source_words})


def _display_texts(text: str) -> list[str]:
    """Split text on whitespace and attach punctuation-only pieces to a neighbouring word.

    Pieces before the first word attach to it; every later one to the word before it.
    """
    pieces = text.split()
    word_positions = _positions_with_words(pieces)
    if not word_positions:
        raise ValueError("The text contains no words.")
    bounds = [0] + word_positions[1:] + [len(pieces)]
    return [" ".join(pieces[begin:end]) for begin, end in pairwise(bounds)]


def _positions_with_words(pieces: list[str]) -> list[int]:
    return [position for position, piece in enumerate(pieces) if tokens(piece)]


def _display_tokens(display_texts: list[str]) -> list[tuple[str, int]]:
    """Word tokens of the display text, each with the index of its display word."""
    return [(token, index) for index, word in enumerate(display_texts) for token in tokens(word)]


def _source_tokens(source_words: list[dict]) -> list[tuple[str, int]]:
    """Word tokens of the source, each with the id of its source word."""
    return [(token, word["word_id"]) for word in source_words for token in tokens(word["text"])]


@dataclass(frozen=True)
class _Opcode:
    """A difflib opcode that turns text tokens into source tokens, resolved to display words and source words.

    ``next_display_index`` is the display word at the opcode's text position, or the last
    display word when the opcode lies behind the end of the text.
    """
    operation: str
    display_indexes: list[int]
    source_ids: list[int]
    next_display_index: int


def _token_opcodes(text_tokens: list[tuple[str, int]], source_tokens: list[tuple[str, int]]) -> list[_Opcode]:
    matcher = difflib.SequenceMatcher(a=[t for t, _ in text_tokens], b=[t for t, _ in source_tokens],
                                      autojunk=False)
    return [_resolve(opcode, text_tokens, source_tokens) for opcode in matcher.get_opcodes()]


def _resolve(opcode: tuple, text_tokens: list[tuple[str, int]], source_tokens: list[tuple[str, int]]) -> _Opcode:
    operation, text_begin, text_end, source_begin, source_end = opcode
    next_token = text_tokens[min(text_begin, len(text_tokens) - 1)]
    return _Opcode(operation=operation,
                   display_indexes=[index for _, index in text_tokens[text_begin:text_end]],
                   source_ids=[word_id for _, word_id in source_tokens[source_begin:source_end]],
                   next_display_index=next_token[1])


class _TokenAlignment:
    """Accumulates per-display-word links while walking the token opcodes."""

    def __init__(self, display_count: int):
        self.source_ids: list[list[int]] = [[] for _ in range(display_count)]
        self.changed = [False] * display_count
        self.deleted_ids: list[tuple[int, int]] = []

    def apply(self, opcode: _Opcode) -> None:
        _OPCODE_HANDLERS[opcode.operation](self, opcode)

    def link(self, index: int, source_ids: list[int], changed: bool) -> None:
        for word_id in source_ids:
            if word_id not in self.source_ids[index]:
                self.source_ids[index].append(word_id)
        self.changed[index] |= changed

    def link_equal(self, opcode: _Opcode) -> None:
        for index, word_id in zip(opcode.display_indexes, opcode.source_ids):
            self.link(index, [word_id], changed=False)

    def link_block(self, opcode: _Opcode) -> None:
        """Link a replaced block: pair token by token when lengths match, else share all ids."""
        if len(opcode.display_indexes) == len(opcode.source_ids):
            for index, word_id in zip(opcode.display_indexes, opcode.source_ids):
                self.link(index, [word_id], changed=True)
            return
        for index in opcode.display_indexes:
            self.link(index, opcode.source_ids, changed=True)

    def link_inserted(self, opcode: _Opcode) -> None:
        """Mark text tokens without a source counterpart as changed."""
        for index in opcode.display_indexes:
            self.link(index, [], changed=True)

    def delete(self, opcode: _Opcode) -> None:
        """Record source tokens missing from the text before the display word at their position."""
        for word_id in opcode.source_ids:
            self.deleted_ids.append((opcode.next_display_index, word_id))

    def display_words(self, display_texts: list[str], source_text_by_id: dict[int, str]) -> list[DisplayWord]:
        removed_before = self._resolve_deletions(len(display_texts))
        return [self._display_word(index, text, removed_before[index], source_text_by_id)
                for index, text in enumerate(display_texts)]

    def _resolve_deletions(self, display_count: int) -> list[dict[int, None]]:
        """A partly deleted source word changes its display word; a wholly deleted one is removed before
        the next display word. Returns the removed source ids per display word, in order and without repeats."""
        removed_before: list[dict[int, None]] = [{} for _ in range(display_count)]
        for next_index, word_id in self.deleted_ids:
            owner = self._owner_of(word_id)
            if owner is None:
                removed_before[next_index][word_id] = None
            else:
                self.changed[owner] = True
        return removed_before

    def _display_word(self, index: int, text: str, removed_ids: dict[int, None],
                      source_text_by_id: dict[int, str]) -> DisplayWord:
        ids = tuple(sorted(self.source_ids[index]))
        return DisplayWord(text=text, source_word_ids=ids, change=self._change_of(index, ids),
                           original_text=" ".join(source_text_by_id[i] for i in ids),
                           removed_before=" ".join(source_text_by_id[i] for i in removed_ids))

    def _change_of(self, index: int, ids: tuple[int, ...]) -> Change:
        if not self.changed[index]:
            return "unchanged"
        return "replaced" if ids else "inserted"

    def _owner_of(self, word_id: int) -> int | None:
        """Return the last display word linked to a partially deleted source word, if any."""
        owners = [index for index, ids in enumerate(self.source_ids) if word_id in ids]
        return owners[-1] if owners else None


# Opcodes turn the text into the source: "delete" is text-only, "insert" is source-only.
_OPCODE_HANDLERS = {"equal": _TokenAlignment.link_equal, "replace": _TokenAlignment.link_block,
                    "delete": _TokenAlignment.link_inserted, "insert": _TokenAlignment.delete}
