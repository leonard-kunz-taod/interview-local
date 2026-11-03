# German fixtures intentionally exercise non-ASCII speech.
import copy
import hashlib
import json
import unittest

from support.fake_ollama import IDENTITY, FakeOllama, ScriptedModel, served_by

from interview_local.core import Turn, Word, merge
from interview_local.llm import enhance, review

REORDERED = "The editor skipped, duplicated or reordered source words."
EDITOR_REASON = "Grouped into a readable paragraph."
INVALID_WORD_RANGES = (
    ("first word as a string", {"first_word": "0"}),
    ("first word as a float", {"first_word": 0.0}),
    ("last word as a float", {"last_word": 1.0}),
    ("first word as a boolean", {"first_word": True}),
    ("last word before the first", {"last_word": -1}),
    ("last word beyond the chunk", {"last_word": 2}),
)
MALFORMED_VERDICTS = (
    ("uncertainty as a string", {"uncertain": "no"}),
    ("reasons as a string", {"reasons": "none"}),
    ("reasons holding a number", {"reasons": [1]}),
    ("paragraph id as a float", {"paragraph_id": 0.0}),
)


def silent(_message):
    pass


def canonical_sha256(value):
    """The fingerprint an auditor recomputes from canonical JSON."""
    canonical = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def two_word_source():
    return merge(
        [Word(0, 0.3, "Hello"), Word(0.3, 0.7, "there.")], [Turn(0, 1, "s")], []
    )


def counted_source(count):
    words = [Word(i * 0.1, i * 0.1 + 0.05, f"w{i}") for i in range(count)]
    return merge(words, [Turn(0, count, "s")], [])


def ten_word_source():
    text = "one two three four five six seven eight nine ten"
    words = [Word(i, i + 0.5, t) for i, t in enumerate(text.split())]
    return merge(words, [Turn(0, 11, "s")], [])


def two_speaker_source():
    return merge(
        [Word(0, 0.3, "Hello"), Word(0.3, 0.7, "there.")],
        [Turn(0, 0.3, "s"), Turn(0.3, 1, "t")],
        [],
    )


def source_with_a_second_speaker():
    """Both words are speaker A's, but speaker B exists in the transcript."""
    return merge(
        [Word(0, 0.3, "Hello"), Word(0.3, 0.7, "there.")],
        [Turn(0, 1, "s"), Turn(1, 2, "t")],
        [],
    )


def timed_source(*spans):
    """One word per (start, end) span, all from one speaker."""
    words = [Word(start, end, f"w{i}") for i, (start, end) in enumerate(spans)]
    return merge(words, [Turn(0, spans[-1][1] + 1, "s")], [])


def edited(source, model=None, status=silent):
    with served_by(FakeOllama(model)):
        return enhance(source, status)


def reviewed(draft, model=None, status=silent):
    with served_by(FakeOllama(model)):
        return review(draft, status)


def rejected_chunks(model, source=None):
    return edited(source or two_word_source(), model)["metadata"]["rejected_chunks"]


def reviewed_paragraph(model):
    return reviewed(edited(two_word_source()), model)["paragraphs"][0]


class EmptyEdit(ScriptedModel):
    def edit(self, words):
        return {"paragraphs": []}


class RejectsTheSecondChunk(ScriptedModel):
    """Edits the first chunk into an unattributed and a short attributed paragraph."""

    def edit(self, words):
        if words[0]["id"] > 0:
            self.invalid_edits = 1
            return super().edit(words)
        head = dict(self.paragraph(words[:110]), speaker=None)
        return {"paragraphs": [head, self.paragraph(words[110:])]}


class DuplicateReview(ScriptedModel):
    def review(self, paragraphs):
        response = super().review(paragraphs)
        response["reviews"] *= 2
        return response


class EditTests(unittest.TestCase):
    def test_editing_leaves_the_source_transcript_unchanged(self):
        source = two_word_source()
        before = copy.deepcopy(source)
        edited(source)
        self.assertEqual(source, before)

    def test_paragraph_references_every_source_word(self):
        ids = edited(two_word_source())["paragraphs"][0]["source_word_ids"]
        self.assertEqual(ids, [0, 1])

    def test_model_is_unloaded_after_editing(self):
        service = FakeOllama()
        with served_by(service):
            enhance(two_word_source(), silent)
        self.assertFalse(service.loaded)

    def test_draft_is_marked_not_reviewed(self):
        verification = edited(two_word_source())["metadata"]["verification"]
        self.assertEqual(verification, "not_reviewed")

    def test_draft_metadata_records_the_editor_and_what_its_flags_mean(self):
        self.assertEqual(
            edited(two_word_source())["metadata"],
            {
                **IDENTITY,
                "rejected_chunks": 0,
                "verification": "not_reviewed",
                "uncertainty": "heuristic flags, not calibrated probabilities",
            },
        )

    def test_draft_declares_its_schema_stage_and_language(self):
        draft = edited(two_word_source())
        self.assertEqual(
            (draft["schema_version"], draft["stage"], draft["language"]),
            (2, "readable", "de"),
        )

    def test_draft_keeps_the_language_of_its_source(self):
        source = dict(two_word_source(), language="en")
        self.assertEqual(edited(source)["language"], "en")

    def test_source_without_a_language_is_taken_as_german(self):
        source = two_word_source()
        del source["language"]
        self.assertEqual(edited(source)["language"], "de")

    def test_source_fingerprint_is_the_sha256_of_its_canonical_json(self):
        source = merge([Word(0, 0.3, "Grüße")], [Turn(0, 1, "s")], [])
        self.assertEqual(edited(source)["source_sha256"], canonical_sha256(source))

    def test_long_source_is_edited_in_chunks_of_120_words(self):
        messages = []
        edited(counted_source(130), status=messages.append)
        self.assertEqual(
            [m for m in messages if m.startswith("LLM readability")],
            [
                "LLM readability: words 1–120 / 130",
                "LLM readability: words 121–130 / 130",
            ],
        )

    def test_valid_first_answer_needs_no_retry(self):
        service = FakeOllama()
        with served_by(service):
            enhance(two_word_source(), silent)
        self.assertEqual(len(service.chat_requests()), 1)

    def test_adjacent_paragraphs_of_one_speaker_are_joined(self):
        result = edited(two_word_source(), ScriptedModel(split=True))
        ids = [p["source_word_ids"] for p in result["paragraphs"]]
        self.assertEqual(ids, [[0, 1]])

    def test_joined_paragraph_combines_wording_sources_and_flags(self):
        source = merge(
            [Word(0, 0.3, "Hello", "estimated"), Word(0.3, 0.7, "there.")],
            [Turn(0, 1, "s")],
            [],
        )
        paragraph = edited(source, ScriptedModel(split=True))["paragraphs"][0]
        self.assertEqual(
            {key: paragraph[key] for key in JOINED_KEYS},
            {
                "text": "Hello there.",
                "source_text": "Hello there.",
                "source_word_ids": [0, 1],
                "start": 0,
                "end": 0.7,
                "uncertain": True,
                "overlap": False,
                "content_changed": False,
                "speaker_changed": False,
                "uncertainty_reasons": ["estimated_or_fallback_timing"],
            },
        )

    def test_joined_paragraph_unites_the_uncertainty_reasons_of_its_parts(self):
        source = merge(
            [Word(0, 0.3, "Hello", "estimated"), Word(0.3, 0.7, "there.")],
            [Turn(0, 1, "s")],
            [Turn(0, 1, "s"), Turn(0.4, 1, "t")],
        )
        paragraph = edited(source, ScriptedModel(split=True))["paragraphs"][0]
        self.assertEqual(
            paragraph["uncertainty_reasons"],
            ["estimated_or_fallback_timing", "overlapping_speech"],
        )

    def test_joined_paragraphs_never_exceed_120_source_words(self):
        result = edited(counted_source(130))
        sizes = [len(p["source_word_ids"]) for p in result["paragraphs"]]
        self.assertEqual(sizes, [120, 10])

    def test_halves_of_a_120_word_chunk_are_joined_back_together(self):
        result = edited(counted_source(120), ScriptedModel(split=True))
        sizes = [len(p["source_word_ids"]) for p in result["paragraphs"]]
        self.assertEqual(sizes, [120])

    def test_paragraphs_paused_two_seconds_apart_stay_separate(self):
        result = edited(timed_source((0, 1), (3, 4)), ScriptedModel(split=True))
        self.assertEqual(len(result["paragraphs"]), 2)

    def test_paragraphs_spanning_sixty_seconds_stay_separate(self):
        result = edited(timed_source((0, 1), (1.5, 60)), ScriptedModel(split=True))
        self.assertEqual(len(result["paragraphs"]), 2)

    def test_nearby_paragraphs_late_in_the_recording_are_joined(self):
        result = edited(timed_source((40, 41), (41.5, 42)), ScriptedModel(split=True))
        self.assertEqual(len(result["paragraphs"]), 1)

    def test_paragraphs_of_a_second_chunk_cover_their_own_source_words(self):
        result = edited(counted_source(130), ScriptedModel(split=True))
        ids = [p["source_word_ids"] for p in result["paragraphs"]]
        self.assertEqual(
            (result["metadata"]["rejected_chunks"], ids),
            (0, [list(range(120)), list(range(120, 130))]),
        )

    def test_speaker_correction_is_flagged(self):
        paragraph = edited(source_with_a_second_speaker(), ScriptedModel(speaker="B"))[
            "paragraphs"
        ][0]
        self.assertEqual(
            (paragraph["speaker_changed"], paragraph["uncertainty_reasons"]),
            (True, ["speaker_changed_by_llm"]),
        )

    def test_wording_change_is_flagged(self):
        changed = "one two three four five six seven eight nine"
        paragraph = edited(ten_word_source(), ScriptedModel(wording=changed))[
            "paragraphs"
        ][0]
        self.assertEqual(
            (paragraph["content_changed"], paragraph["uncertainty_reasons"]),
            (True, ["wording_changed_by_llm"]),
        )

    def test_edit_keeping_exactly_85_percent_of_the_wording_is_accepted(self):
        kept = " ".join(f"w{i}" for i in range(17))
        result = edited(counted_source(20), ScriptedModel(wording=kept + " x y z"))
        self.assertEqual(
            (result["metadata"]["rejected_chunks"], result["paragraphs"][0]["text"]),
            (0, kept + " x y z"),
        )

    def test_lightly_edited_repetitive_speech_is_accepted(self):
        words = [Word(i * 0.1, i * 0.1 + 0.05, "ja-also") for i in range(120)]
        source = merge(words, [Turn(0, 20, "s")], [])
        hesitation_added = " ".join(["ja-also"] * 60 + ["ähm"] + ["ja-also"] * 60)
        model = ScriptedModel(wording=hesitation_added)
        self.assertEqual(rejected_chunks(model, source), 0)

    def test_accepted_paragraph_keeps_the_editor_reason(self):
        paragraph = edited(two_word_source())["paragraphs"][0]
        self.assertEqual(paragraph["editor_reason"], EDITOR_REASON)

    def test_paragraph_spans_the_time_of_its_source_words(self):
        paragraph = edited(two_word_source())["paragraphs"][0]
        self.assertEqual((paragraph["start"], paragraph["end"]), (0, 0.7))

    def test_unchanged_paragraph_is_certain(self):
        paragraph = edited(two_word_source())["paragraphs"][0]
        self.assertEqual(
            (paragraph["uncertain"], paragraph["uncertainty_reasons"]), (False, [])
        )

    def test_source_uncertainty_reasons_carry_over_to_the_paragraph(self):
        source = merge(
            [Word(0, 0.3, "Hello", "estimated"), Word(0.3, 0.7, "there.")],
            [Turn(0, 1, "s")],
            [],
        )
        paragraph = edited(source)["paragraphs"][0]
        self.assertEqual(
            (paragraph["uncertain"], paragraph["uncertainty_reasons"]),
            (True, ["estimated_or_fallback_timing"]),
        )

    def test_overlapping_source_speech_marks_the_paragraph(self):
        source = merge(
            [Word(0, 0.3, "Hello"), Word(0.3, 0.7, "there.")],
            [Turn(0, 1, "s")],
            [Turn(0, 1, "s"), Turn(0.5, 1, "t")],
        )
        paragraph = edited(source)["paragraphs"][0]
        self.assertEqual((paragraph["overlap"], paragraph["uncertain"]), (True, True))

    def test_words_without_ids_are_numbered_in_order(self):
        source = two_word_source()
        for word in source["words"]:
            del word["word_id"]
        ids = edited(source)["paragraphs"][0]["source_word_ids"]
        self.assertEqual(ids, [0, 1])

    def test_legacy_words_without_speaker_fields_are_unassigned_and_certain(self):
        source = two_word_source()
        source["words"] = [
            {key: w[key] for key in ("start", "end", "text")} for w in source["words"]
        ]
        paragraph = edited(source, ScriptedModel(speaker=None))["paragraphs"][0]
        self.assertEqual(
            (paragraph["speaker"], paragraph["uncertain"], paragraph["overlap"]),
            (None, False, False),
        )

    def test_non_consecutive_source_word_ids_are_rejected(self):
        source = two_word_source()
        source["words"][0]["word_id"] = 1
        with self.assertRaises(ValueError):
            edited(source)

    def test_paragraphs_are_numbered_in_order(self):
        result = edited(two_speaker_source(), ScriptedModel(invalid_edits=2))
        self.assertEqual([p["paragraph_id"] for p in result["paragraphs"]], [0, 1])


JOINED_KEYS = (
    "text",
    "source_text",
    "source_word_ids",
    "start",
    "end",
    "uncertain",
    "overlap",
    "content_changed",
    "speaker_changed",
    "uncertainty_reasons",
)


class RejectedEditTests(unittest.TestCase):
    def test_invalid_edit_is_retried_once(self):
        result = edited(two_word_source(), ScriptedModel(invalid_edits=1))
        self.assertEqual(result["metadata"]["rejected_chunks"], 0)

    def test_retry_tells_the_editor_why_its_answer_was_rejected(self):
        service = FakeOllama(ScriptedModel(invalid_edits=1))
        with served_by(service):
            enhance(two_word_source(), silent)
        retry = service.chat_requests()[1]["body"]["messages"][1]["content"]
        self.assertTrue(json.loads(retry)["validation_error"].startswith(REORDERED))

    def test_chunk_rejected_twice_is_counted(self):
        result = edited(two_word_source(), ScriptedModel(invalid_edits=2))
        self.assertEqual(result["metadata"]["rejected_chunks"], 1)

    def test_every_rejected_chunk_is_counted(self):
        result = edited(counted_source(130), ScriptedModel(invalid_edits=4))
        self.assertEqual(result["metadata"]["rejected_chunks"], 2)

    def test_rejection_is_reported_with_its_reason(self):
        messages = []
        edited(two_word_source(), ScriptedModel(invalid_edits=2), messages.append)
        self.assertTrue(any(REORDERED in message for message in messages))

    def test_chunk_rejected_twice_retains_original_wording(self):
        result = edited(two_word_source(), ScriptedModel(invalid_edits=2))
        self.assertEqual(result["paragraphs"][0]["text"], "Hello there.")

    def test_rejected_chunk_explains_why_its_wording_was_kept(self):
        paragraph = edited(two_word_source(), ScriptedModel(invalid_edits=2))[
            "paragraphs"
        ][0]
        self.assertEqual(
            paragraph["editor_reason"],
            "Editor response rejected; original wording retained.",
        )

    def test_rejected_word_without_text_is_shown_as_an_ellipsis(self):
        source = merge([Word(0, 0.3, " ")], [Turn(0, 1, "s")], [])
        paragraph = edited(source, ScriptedModel(invalid_edits=2))["paragraphs"][0]
        self.assertEqual(paragraph["text"], "…")

    def test_chunk_rejected_twice_is_flagged_uncertain(self):
        paragraph = edited(two_word_source(), ScriptedModel(invalid_edits=2))[
            "paragraphs"
        ][0]
        self.assertEqual(
            (paragraph["uncertain"], paragraph["uncertainty_reasons"]),
            (True, ["editor_output_rejected"]),
        )

    def test_invented_wording_is_rejected(self):
        model = ScriptedModel(wording="Invented statement")
        result = edited(two_word_source(), model)
        self.assertEqual(result["metadata"]["rejected_chunks"], 1)

    def test_unknown_speaker_from_the_editor_is_rejected(self):
        result = edited(two_word_source(), ScriptedModel(speaker="Z"))
        self.assertEqual(result["metadata"]["rejected_chunks"], 1)

    def test_empty_paragraph_list_is_rejected(self):
        self.assertEqual(rejected_chunks(EmptyEdit()), 1)

    def test_paragraphs_that_are_not_a_list_are_rejected(self):
        model = ScriptedModel(response={"paragraphs": "Hello there."})
        self.assertEqual(rejected_chunks(model), 1)

    def test_blank_edited_text_is_rejected(self):
        self.assertEqual(rejected_chunks(ScriptedModel(wording="  ")), 1)

    def test_missing_edited_text_is_rejected(self):
        model = ScriptedModel(paragraph_fields={"text": None})
        self.assertEqual(rejected_chunks(model), 1)

    def test_non_boolean_uncertainty_is_rejected(self):
        model = ScriptedModel(paragraph_fields={"uncertain": "no"})
        self.assertEqual(rejected_chunks(model), 1)

    def test_missing_editorial_reason_is_rejected(self):
        model = ScriptedModel(paragraph_fields={"reason": None})
        self.assertEqual(rejected_chunks(model), 1)

    def test_word_range_outside_the_chunk_order_is_rejected(self):
        for case, fields in INVALID_WORD_RANGES:
            with self.subTest(case):
                model = ScriptedModel(paragraph_fields=fields)
                paragraph = edited(two_word_source(), model)["paragraphs"][0]
                self.assertEqual(paragraph["editor_error"], REORDERED)

    def test_omitted_trailing_words_are_rejected(self):
        model = ScriptedModel(paragraph_fields={"last_word": 0, "text": "Hello"})
        paragraph = edited(two_word_source(), model)["paragraphs"][0]
        self.assertEqual(
            paragraph["editor_error"],
            "The editor omitted words at the end of the chunk.",
        )

    def test_rejected_chunk_keeps_one_paragraph_per_speaker_run(self):
        result = edited(two_speaker_source(), ScriptedModel(invalid_edits=2))
        paragraphs = [(p["speaker"], p["text"]) for p in result["paragraphs"]]
        self.assertEqual(paragraphs, [("A", "Hello"), ("B", "there.")])

    def test_rejected_chunk_records_the_last_editor_error(self):
        paragraph = edited(two_word_source(), ScriptedModel(invalid_edits=2))[
            "paragraphs"
        ][0]
        self.assertEqual(paragraph["editor_error"], REORDERED)

    def test_joined_paragraph_keeps_the_error_of_its_rejected_part(self):
        source = counted_source(130)
        paragraph = edited(source, RejectsTheSecondChunk())["paragraphs"][-1]
        self.assertEqual(
            (paragraph["source_word_ids"], paragraph["editor_error"]),
            (list(range(110, 130)), REORDERED),
        )


class ReviewTests(unittest.TestCase):
    def test_review_never_erases_source_uncertainty(self):
        source = two_word_source()
        source["words"][0]["uncertain"] = True
        checked = reviewed(edited(source))
        self.assertTrue(checked["paragraphs"][0]["uncertain"])

    def test_review_records_the_reviewer_verdict(self):
        checked = reviewed(edited(two_word_source()))
        paragraph = checked["paragraphs"][0]
        self.assertEqual(
            (
                paragraph["review_status"],
                paragraph["review_uncertain"],
                paragraph["review_reasons"],
            ),
            ("compared", False, []),
        )

    def test_review_is_marked_as_source_comparison_only(self):
        checked = reviewed(edited(two_word_source()))
        self.assertEqual(
            checked["metadata"]["verification"],
            "source_comparison_only_not_audio_verified",
        )

    def test_review_progress_counts_the_paragraphs(self):
        draft = edited(two_speaker_source(), ScriptedModel(invalid_edits=2))
        messages = []
        reviewed(draft, status=messages.append)
        self.assertEqual(
            messages,
            [
                "LLM comparison review: paragraph 1 / 2",
                "LLM comparison review: paragraph 2 / 2",
            ],
        )

    def test_invalid_review_response_requires_manual_review(self):
        model = ScriptedModel(review_id_offset=1)
        checked = reviewed(edited(two_word_source()), model)
        paragraph = checked["paragraphs"][0]
        self.assertEqual(
            (
                paragraph["review_status"],
                paragraph["review_uncertain"],
                paragraph["review_reasons"],
            ),
            (
                "invalid_response",
                True,
                ["Invalid model review; manual review required."],
            ),
        )

    def test_wording_change_stays_uncertain_when_reviewer_agrees(self):
        changed = "one two three four five six seven eight nine"
        draft = edited(ten_word_source(), ScriptedModel(wording=changed))
        checked = reviewed(draft)
        self.assertTrue(checked["paragraphs"][0]["uncertain"])

    def test_review_rederives_a_wording_change_the_draft_lost(self):
        changed = "one two three four five six seven eight nine"
        draft = edited(ten_word_source(), ScriptedModel(wording=changed))
        draft["paragraphs"][0]["content_changed"] = False
        self.assertTrue(reviewed(draft)["paragraphs"][0]["content_changed"])

    def test_review_rederives_a_speaker_change_the_draft_lost(self):
        draft = edited(source_with_a_second_speaker(), ScriptedModel(speaker="B"))
        draft["paragraphs"][0]["speaker_changed"] = False
        self.assertTrue(reviewed(draft)["paragraphs"][0]["speaker_changed"])

    def test_review_with_malformed_verdict_requires_manual_review(self):
        for case, fields in MALFORMED_VERDICTS:
            with self.subTest(case):
                paragraph = reviewed_paragraph(ScriptedModel(review_fields=fields))
                self.assertEqual(paragraph["review_status"], "invalid_response")

    def test_more_than_one_verdict_per_paragraph_requires_manual_review(self):
        paragraph = reviewed_paragraph(DuplicateReview())
        self.assertEqual(paragraph["review_status"], "invalid_response")

    def test_reviewer_reasons_are_limited_to_ten(self):
        reasons = [f"reason {i}" for i in range(12)]
        paragraph = reviewed_paragraph(
            ScriptedModel(review_fields={"reasons": reasons})
        )
        self.assertEqual(paragraph["review_reasons"], reasons[:10])

    def test_reviewer_doubt_marks_the_paragraph_uncertain(self):
        model = ScriptedModel(review_fields={"uncertain": True})
        self.assertTrue(reviewed_paragraph(model)["uncertain"])

    def test_agreeing_review_keeps_a_clean_paragraph_certain(self):
        self.assertFalse(reviewed_paragraph(ScriptedModel())["uncertain"])

    def test_review_records_the_reviewed_edit_and_reviewer(self):
        draft = edited(two_word_source())
        metadata = reviewed(draft)["metadata"]
        self.assertEqual(
            (metadata["reviewer"], metadata["reviewed_edit_sha256"]),
            (IDENTITY, canonical_sha256(draft)),
        )

    def test_model_is_unloaded_after_reviewing(self):
        draft = edited(two_word_source())
        service = FakeOllama()
        with served_by(service):
            review(draft, silent)
        self.assertFalse(service.loaded)

    def test_review_rejects_a_missing_embedded_source(self):
        draft = edited(two_word_source())
        del draft["source_transcript"]
        with self.assertRaises(ValueError):
            reviewed(draft)

    def test_review_requires_a_readable_stage(self):
        with self.assertRaises(ValueError):
            reviewed({"stage": "speakers"})

    def test_review_rejects_a_modified_embedded_source(self):
        draft = edited(two_word_source())
        draft["source_transcript"]["words"][0]["text"] = "changed"
        with self.assertRaises(ValueError):
            reviewed(draft)

    def test_review_rejects_a_source_whose_speaker_names_changed(self):
        draft = edited(two_word_source())
        draft["source_transcript"]["speakers"]["A"] = "Mallory"
        with self.assertRaises(ValueError):
            reviewed(draft)

    def test_review_rejects_duplicate_word_coverage(self):
        draft = edited(two_word_source())
        draft["paragraphs"][0]["source_word_ids"] = [0, 0]
        with self.assertRaises(ValueError):
            reviewed(draft)

    def test_review_rejects_a_paragraph_whose_source_text_changed(self):
        draft = edited(two_word_source())
        draft["paragraphs"][0]["source_text"] = "Hello here."
        with self.assertRaises(ValueError):
            reviewed(draft)


if __name__ == "__main__":
    unittest.main()
