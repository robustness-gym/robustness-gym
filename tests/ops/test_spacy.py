from unittest import TestCase

from robustnessgym import lookup
from robustnessgym.ops import SpacyOp
from tests.testbeds import MockTestBedv0


class TestSpacy(TestCase):
    def setUp(self):
        self.testbed = MockTestBedv0()

    def test_apply(self):
        # Create the Spacy cached operation
        spacy = SpacyOp()

        # Apply it
        dataset = spacy(self.testbed.dataset, ["text"])
        print(dataset.column_names)

        # Retrieve information to test
        sentences = [doc.sents for doc in lookup(dataset, spacy, ["text"])]
        tokens = [list(doc) for doc in lookup(dataset, spacy, ["text"])]
        entities = [doc.ents for doc in lookup(dataset, spacy, ["text"])]
        num_tokens = [len(list(doc)) for doc in lookup(dataset, spacy, ["text"])]

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
