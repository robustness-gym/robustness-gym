from robustnessgym.core.tools import recmerge
from unittest import TestCase


class TestTools(TestCase):

    def test_recmerge(self):
        output = recmerge(
            {'a': 2, 'b': 3, 'd': {'e': [1, 2, 3], 'f': [3, 4, 5]}, 'g': 17},
            {'b': 12, 'd': {'e': [1, 2, 3], 'f': [3, 4]}},
            {'a': 4, 'd': {'f': [3]}},
        )
        self.assertEqual(output, {'a': 4, 'b': 12, 'd': {'e': [1, 2, 3], 'f': [3]}, 'g': 17})

        output = recmerge(
            {'a': 2, 'b': 3, 'd': {'e': [1, 2, 3], 'f': [3, 4, 5]}, 'g': 17},
            {'b': 12, 'd': {'e': [1, 2, 3], 'f': [3, 4]}},
            {'a': 4, 'd': {'f': [3]}},
            merge_sequences=True
        )
        self.assertEqual(output, {'a': 4, 'b': 12, 'd': {'e': [1, 2, 3, 1, 2, 3], 'f': [3, 4, 5, 3, 4, 3]}, 'g': 17})
