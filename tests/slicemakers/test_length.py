import unittest
import robustness_gym as rg


class LengthTest(unittest.TestCase):
    def test_min_length(self):
        dataset = rg.Dataset.from_batch({'text': ['the quick brown fox jumps',
                                                  'the quick brown',
                                                  'the quick brown fox jumps over the lazy dogs']})
        dataset = rg.stow(dataset=dataset, cached_ops={rg.Spacy(): [['text']]})
        slicemaker = rg.MinLength([5, 1, 7])
        new_dataset, slices, slice_membership = slicemaker(dataset, keys=['text'], store_compressed=False)
        self.assertEqual(
            slice_membership.tolist(),
            [
                [1, 1, 0],
                [0, 1, 0],
                [1, 1, 1]
            ]
        )


if __name__ == '__main__':
    unittest.main()
