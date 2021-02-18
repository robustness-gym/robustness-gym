from unittest import TestCase

from robustnessgym.cachedops.stanza import Stanza
from tests.testbeds import MockTestBedv0


class TestStanza(TestCase):
    def setUp(self):
        self.testbed = MockTestBedv0()

    def test_apply(self):
        # Create the Stanza cached operation
        stanza = Stanza()
        dataset = stanza(self.testbed.dataset, columns=["text"])

        # Make sure things match up
        self.assertEqual(
            stanza.retrieve(
                dataset[:],
                ["text"],
                proc_fns=lambda decoded_batch: [
                    doc.get("lemma") for doc in decoded_batch
                ],
            ),
            [
                ["the", "man", "be", "walk", "."],
                ["the", "man", "be", "run", "."],
                ["the", "woman", "be", "sprint", "."],
                ["the", "woman", "be", "rest", "."],
                ["the", "hobbit", "be", "fly", "."],
                ["the", "hobbit", "be", "swim", "."],
            ],
        )
