from unittest import TestCase

from robustnessgym.cachedops import Spacy
from tests.testbeds import MockTestBedv0


class TestSpacy(TestCase):
    def setUp(self):
        self.testbed = MockTestBedv0()

    def test_apply(self):
        # Create the Spacy cached operation
        spacy = Spacy()

        # Apply it
        dataset = spacy(self.testbed.dataset, ["text"])
        print(dataset.column_names)

        # Retrieve information to test
        sentences = spacy.retrieve(dataset[:], ["text"], proc_fns=spacy.sentences)
        tokens = spacy.retrieve(dataset[:], ["text"], proc_fns=spacy.tokens)
        entities = spacy.retrieve(dataset[:], ["text"], proc_fns=spacy.entities)
        num_tokens = spacy.retrieve(dataset[:], ["text"], proc_fns=spacy.num_tokens)

        self.assertEqual(
            sentences,
            [
                ["The man is walking."],
                ["The man is running."],
                ["The woman is sprinting."],
                ["The woman is resting."],
                ["The hobbit is flying."],
                ["The hobbit is swimming."],
            ],
        )

        self.assertEqual(
            tokens,
            [
                ["The", "man", "is", "walking", "."],
                ["The", "man", "is", "running", "."],
                ["The", "woman", "is", "sprinting", "."],
                ["The", "woman", "is", "resting", "."],
                ["The", "hobbit", "is", "flying", "."],
                ["The", "hobbit", "is", "swimming", "."],
            ],
        )

        self.assertEqual(entities, [[], [], [], [], [], []])
        self.assertEqual(num_tokens, [5, 5, 5, 5, 5, 5])
