from unittest import TestCase

from robustnessgym.core.operation import lookup
from robustnessgym.ops.stanza import StanzaOp
from tests.testbeds import MockTestBedv0


class TestStanza(TestCase):
    def setUp(self):
        self.testbed = MockTestBedv0()

    def test_apply(self):
        # Create the Stanza cached operation
        stanza = StanzaOp()
        dataset = stanza(self.testbed.dataset, columns=["text"])

        # Make sure things match up
        self.assertEqual(
            [
                doc.get("lemma")
                for doc in lookup(
                    dataset,
                    stanza,
                    ["text"],
                )
            ],
            [
                ["the", "man", "be", "walk", "."],
                ["the", "man", "be", "run", "."],
                ["the", "woman", "be", "sprint", "."],
                ["the", "woman", "be", "rest", "."],
                ["the", "hobbit", "be", "fly", "."],
                ["the", "hobbit", "be", "swim", "."],
            ],
        )
