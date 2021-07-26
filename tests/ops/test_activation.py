from unittest import TestCase

import torch
import torch.nn as nn

from robustnessgym.core.operation import lookup
from robustnessgym.ops.activation import ActivationOp
from tests.testbeds import MockVisionTestBed


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Conv2d(
            in_channels=3, out_channels=2, kernel_size=1, bias=False
        )
        self.hidden.weight.data[:] = torch.zeros_like(self.hidden.weight.data)

    def forward(self, x):
        x = x.permute(0, -1, 1, 2).to(torch.float)
        x = self.hidden(x)
        return x


class TestActivation(TestCase):
    def setUp(self):
        self.model = TestModel()
        self.dataset = MockVisionTestBed(wrap_dataset=True).dataset

    def test_apply(self):
        op = ActivationOp(model=self.model, target_module="hidden")

        dataset = op(self.dataset, columns=["i"])

        # Make sure things match up
        acts = lookup(dataset, op, ["i"])
        self.assertEqual(type(acts), list)

        acts = torch.stack(acts)
        self.assertTrue(torch.all(torch.eq(acts, 0)))
        self.assertEqual(list(acts.shape), [4, 2, 10, 10])
