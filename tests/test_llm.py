import copy
import unittest

from fake_ollama import FakeClient

from interview_local.core import Turn, Word, merge
from interview_local.llm import enhance, review


def silent(_message):
    pass


def two_word_source():
    return merge([Word(0, 0.3, "Hello"), Word(0.3, 0.7, "there.")], [Turn(0, 1, "s")])


def counted_source(count):
    words = [Word(i * 0.1, i * 0.1 + 0.05, f"w{i}") for i in range(count)]
    return merge(words, [Turn(0, count, "s")])


def ten_word_source():
    text = "one two three four five six seven eight nine ten"
    words = [Word(i, i + 0.5, t) for i, t in enumerate(text.split())]
    return merge(words, [Turn(0, 11, "s")])


def edited(source, client=None):
    return enhance(source, client=client or FakeClient(), status=silent)


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
        client = FakeClient()
        edited(two_word_source(), client)
        self.assertTrue(client.unloaded)

    def test_draft_is_marked_not_reviewed(self):
        verification = edited(two_word_source())["metadata"]["verification"]
        self.assertEqual(verification, "not_reviewed")

    def test_long_source_is_edited_in_chunks_of_120_words(self):
        messages = []
        enhance(counted_source(130), client=FakeClient(), status=messages.append)
        self.assertEqual(
            [m for m in messages if m.startswith("LLM readability")],
            [
                "LLM readability: words 1–120 / 130",
                "LLM readability: words 121–130 / 130",
            ],
        )

    def test_adjacent_paragraphs_of_one_speaker_are_joined(self):
        result = edited(two_word_source(), FakeClient(split=True))
        ids = [p["source_word_ids"] for p in result["paragraphs"]]
        self.assertEqual(ids, [[0, 1]])

    def test_joined_paragraphs_never_exceed_120_source_words(self):
        result = edited(counted_source(130))
        sizes = [len(p["source_word_ids"]) for p in result["paragraphs"]]
        self.assertEqual(sizes, [120, 10])

    def test_speaker_correction_is_flagged(self):
        source = merge(
            [Word(0, 0.3, "Hello"), Word(0.3, 0.7, "there.")],
            [Turn(0, 1, "s"), Turn(1, 2, "t")],
        )
        paragraph = edited(source, FakeClient(speaker="B"))["paragraphs"][0]
        self.assertEqual(
            (paragraph["speaker_changed"], paragraph["uncertainty_reasons"]),
            (True, ["speaker_changed_by_llm"]),
        )

    def test_wording_change_is_flagged(self):
        changed = "one two three four five six seven eight nine"
        result = edited(ten_word_source(), FakeClient(wording=changed))
        self.assertTrue(result["paragraphs"][0]["content_changed"])


class RejectedEditTests(unittest.TestCase):
    def test_invalid_edit_is_retried_once(self):
        result = edited(two_word_source(), FakeClient(invalid_edits=1))
        self.assertEqual(result["metadata"]["rejected_chunks"], 0)

    def test_chunk_rejected_twice_is_counted(self):
        result = edited(two_word_source(), FakeClient(invalid_edits=2))
        self.assertEqual(result["metadata"]["rejected_chunks"], 1)

    def test_chunk_rejected_twice_retains_original_wording(self):
        result = edited(two_word_source(), FakeClient(invalid_edits=2))
        self.assertEqual(result["paragraphs"][0]["text"], "Hello there.")

    def test_chunk_rejected_twice_is_flagged_uncertain(self):
        paragraph = edited(two_word_source(), FakeClient(invalid_edits=2))[
            "paragraphs"
        ][0]
        self.assertEqual(
            (paragraph["uncertain"], paragraph["uncertainty_reasons"]),
            (True, ["editor_output_rejected"]),
        )

    def test_invented_wording_is_rejected(self):
        client = FakeClient(wording="Invented statement")
        result = edited(two_word_source(), client)
        self.assertEqual(result["metadata"]["rejected_chunks"], 1)

    def test_unknown_speaker_from_the_editor_is_rejected(self):
        result = edited(two_word_source(), FakeClient(speaker="Z"))
        self.assertEqual(result["metadata"]["rejected_chunks"], 1)


class ReviewTests(unittest.TestCase):
    def test_review_never_erases_source_uncertainty(self):
        source = two_word_source()
        source["words"][0]["uncertain"] = True
        checked = review(edited(source), client=FakeClient(), status=silent)
        self.assertTrue(checked["paragraphs"][0]["uncertain"])

    def test_review_records_the_reviewer_verdict(self):
        checked = review(edited(two_word_source()), FakeClient(), status=silent)
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
        checked = review(edited(two_word_source()), FakeClient(), status=silent)
        self.assertEqual(
            checked["metadata"]["verification"],
            "source_comparison_only_not_audio_verified",
        )

    def test_invalid_review_response_requires_manual_review(self):
        client = FakeClient(review_id_offset=1)
        checked = review(edited(two_word_source()), client, status=silent)
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
        draft = edited(ten_word_source(), FakeClient(wording=changed))
        checked = review(draft, client=FakeClient(), status=silent)
        self.assertTrue(checked["paragraphs"][0]["uncertain"])

    def test_review_requires_a_readable_stage(self):
        with self.assertRaises(ValueError):
            review({"stage": "speakers"}, client=FakeClient(), status=silent)

    def test_review_rejects_a_modified_embedded_source(self):
        draft = edited(two_word_source())
        draft["source_transcript"]["words"][0]["text"] = "changed"
        with self.assertRaises(ValueError):
            review(draft, client=FakeClient(), status=silent)

    def test_review_rejects_duplicate_word_coverage(self):
        draft = edited(two_word_source())
        draft["paragraphs"][0]["source_word_ids"] = [0, 0]
        with self.assertRaises(ValueError):
            review(draft, client=FakeClient(), status=silent)

    def test_review_rejects_a_paragraph_whose_source_text_changed(self):
        draft = edited(two_word_source())
        draft["paragraphs"][0]["source_text"] = "Hello here."
        with self.assertRaises(ValueError):
            review(draft, client=FakeClient(), status=silent)


if __name__ == "__main__":
    unittest.main()
