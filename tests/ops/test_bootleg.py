import unittest
from unittest import TestCase

from robustnessgym.core.operation import lookup
from robustnessgym.ops.bootleg import BootlegAnnotatorOp
from tests.testbeds import MockTestBedv0


class TestBootleg(TestCase):
    def setUp(self):
        self.testbed = MockTestBedv0()
        self.cache_dir = "projects/bootleg/tutorial_data"

    @unittest.skip("Need sufficient disk/memory/GPU to run this test")
    def test_apply(self):
        # Create the Bootleg cached operation
        bootleg = BootlegAnnotatorOp(cache_dir=self.cache_dir)

        dataset = bootleg(self.testbed.dataset, columns=["text"])

        # Make sure things match up
        res = lookup(dataset, bootleg, ["text"])
        bootleg_keys = [
            "qids",
            "probs",
            "titles",
            "cands",
            "cand_probs",
            "spans",
            "aliases",
        ]
        for output in res:
            for k in bootleg_keys:
                assert k in output
