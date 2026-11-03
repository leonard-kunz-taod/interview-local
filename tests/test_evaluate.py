# German fixtures intentionally exercise non-ASCII speech and speaker names.
import unittest
from interview_local.evaluate import distance, normalize

class EvaluationTests(unittest.TestCase):
    def test_normalization_and_edit_distance(self):
        self.assertEqual(normalize('Grüße, {laugh} Äpfel!'), ['grüße','äpfel'])
        self.assertEqual(distance(['a','b','c'],['a','d','c','e']),2)
        self.assertEqual(distance([],['a']),1)
        self.assertEqual(distance(['a'],[]),1)
        self.assertEqual(distance(['a'],['a']),0)
