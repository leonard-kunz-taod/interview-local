import unittest

from interview_local.llm import tokens
from interview_local.word_alignment import align_text_to_source


def source(*texts):
    return [dict(word_id=10 + i, text=text) for i, text in enumerate(texts)]


def is_highlighted(display_words):
    return any(w.change != "unchanged" or w.removed_before for w in display_words)


class AlignTextToSourceTests(unittest.TestCase):
    def test_identical_words_are_unchanged_and_linked_to_their_source(self):
        words = align_text_to_source("Guten Tag", source("Guten", "Tag"))
        self.assertEqual([(w.text, w.source_word_ids, w.change) for w in words],
                         [("Guten", (10,), "unchanged"), ("Tag", (11,), "unchanged")])

    def test_case_and_punctuation_edits_are_not_changes(self):
        words = align_text_to_source("Guten Tag, wie geht's?", source("guten", "tag", "wie", "geht's"))
        self.assertFalse(is_highlighted(words))

    def test_replaced_word_keeps_its_source_and_original_text(self):
        words = align_text_to_source("Guten Morgen", source("Guten", "Tag"))
        self.assertEqual(words[1].change, "replaced")
        self.assertEqual(words[1].source_word_ids, (11,))
        self.assertEqual(words[1].original_text, "Tag")

    def test_inserted_word_has_no_source(self):
        words = align_text_to_source("Guten lieben Tag", source("Guten", "Tag"))
        self.assertEqual(words[1].change, "inserted")
        self.assertEqual(words[1].source_word_ids, ())
        self.assertEqual(words[1].original_text, "")
        self.assertEqual(words[2].change, "unchanged")

    def test_deleted_word_is_reported_before_the_next_word(self):
        words = align_text_to_source("Guten Tag", source("Guten", "äh", "Tag"))
        self.assertEqual(words[1].removed_before, "äh")
        self.assertEqual([w.change for w in words], ["unchanged", "unchanged"])

    def test_trailing_deletion_goes_to_the_last_word(self):
        words = align_text_to_source("Guten Tag", source("Guten", "Tag", "ja"))
        self.assertEqual(words[-1].removed_before, "ja")

    def test_partly_deleted_source_word_marks_its_display_word_replaced(self):
        words = align_text_to_source("z Beispiel", source("z.B.", "Beispiel"))
        self.assertEqual(words[0].change, "replaced")
        self.assertEqual(words[0].original_text, "z.B.")

    def test_punctuation_only_word_attaches_to_the_previous_word(self):
        words = align_text_to_source("Ja – gut", source("Ja", "gut"))
        self.assertEqual([w.text for w in words], ["Ja –", "gut"])

    def test_leading_punctuation_attaches_to_the_first_word(self):
        words = align_text_to_source("– Ja", source("Ja"))
        self.assertEqual([w.text for w in words], ["– Ja"])

    def test_nothing_is_highlighted_exactly_when_content_is_unchanged(self):
        cases = [("Guten Tag.", ["guten", "Tag"]), ("Guten Morgen", ["Guten", "Tag"]),
                 ("Guten Tag", ["Guten", "äh", "Tag"]), ("Guten Tag ja", ["Guten", "Tag"]),
                 ("Hausboot", ["Haus", "boot"]), ("Haus boot", ["Hausboot"]), ("z.B. so", ["z", "B", "so"]),
                 ("Tag Guten", ["Guten", "Tag"]), ("geht's", ["geht", "s"])]
        for text, texts in cases:
            with self.subTest(text=text):
                content_changed = tokens(text) != tokens(" ".join(texts))
                self.assertEqual(is_highlighted(align_text_to_source(text, source(*texts))), content_changed)

    def test_text_without_words_is_rejected(self):
        with self.assertRaises(ValueError):
            align_text_to_source(" – ", source("Ja"))

    def test_empty_source_is_rejected(self):
        with self.assertRaises(ValueError):
            align_text_to_source("Ja", [])


if __name__ == "__main__":
    unittest.main()
