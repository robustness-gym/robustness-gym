"""
Unittests for decorators.
"""
from unittest import TestCase

from robustnessgym.core.decorators import singlecolumn


class TestDecorators(TestCase):

    def test_singlecolumn(self):
        @singlecolumn
        def apply(self, batch, columns):
            print(columns)

        apply(None, None, ['abc'])
        with self.assertRaises(AssertionError):
            apply(None, None, ['abc', 'bcd'])
