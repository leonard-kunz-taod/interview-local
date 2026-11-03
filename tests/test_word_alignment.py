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
        self.assertEqual((words[1].change, words[1].source_word_ids, words[1].original_text),
                         ("replaced", (11,), "Tag"))

    def test_inserted_word_has_no_source_and_leaves_its_neighbours_unchanged(self):
        words = align_text_to_source("Guten lieben Tag", source("Guten", "Tag"))
        self.assertEqual([(w.change, w.source_word_ids, w.original_text) for w in words],
                         [("unchanged", (10,), "Guten"), ("inserted", (), ""), ("unchanged", (11,), "Tag")])

    def test_deleted_word_is_reported_before_the_next_word(self):
        words = align_text_to_source("Guten Tag", source("Guten", "äh", "Tag"))
        self.assertEqual([(w.change, w.removed_before) for w in words], [("unchanged", ""), ("unchanged", "äh")])

    def test_trailing_deletion_goes_to_the_last_word(self):
        words = align_text_to_source("Guten Tag", source("Guten", "Tag", "ja"))
        self.assertEqual(words[-1].removed_before, "ja")

    def test_partly_deleted_source_word_marks_its_display_word_replaced(self):
        words = align_text_to_source("z Beispiel", source("z.B.", "Beispiel"))
        self.assertEqual((words[0].change, words[0].original_text), ("replaced", "z.B."))

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

    def test_display_word_covering_several_source_words_shows_them_joined(self):
        words = align_text_to_source("Hausboot", source("Haus", "boot"))
        self.assertEqual((words[0].source_word_ids, words[0].original_text), ((10, 11), "Haus boot"))

    def test_several_removed_words_are_reported_together(self):
        words = align_text_to_source("Guten Tag", source("Guten", "äh", "ähm", "Tag"))
        self.assertEqual(words[1].removed_before, "äh ähm")

    def test_display_word_with_one_changed_part_is_replaced(self):
        words = align_text_to_source("Haus-Boot", source("Hof", "Boot"))
        self.assertEqual(words[0].change, "replaced")

    def test_frequent_word_between_edited_neighbours_in_a_long_paragraph_stays_unchanged(self):
        texts = []
        for number in range(200):
            texts.append(f"wort{number}")
            if number % 20 == 19:
                texts.append("und")
        first_und = texts.index("und")
        edited = texts[:first_und - 1] + ["neu", "und", "neu"] + texts[first_und + 2:]
        words = align_text_to_source(" ".join(edited), source(*texts))
        self.assertEqual((words[first_und].text, words[first_und].change), ("und", "unchanged"))

    def test_text_without_words_is_rejected(self):
        with self.assertRaises(ValueError):
            align_text_to_source(" – ", source("Ja"))

    def test_empty_source_is_rejected(self):
        with self.assertRaises(ValueError):
            align_text_to_source("Ja", [])


if __name__ == "__main__":
    unittest.main()
