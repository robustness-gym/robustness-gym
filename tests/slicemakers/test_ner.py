import unittest
import robustness_gym as rg


class NERTest(unittest.TestCase):
    def test_entity_frequency(self):
        dataset = rg.Dataset.from_batch({'text': ['Laura Linney stars in the show, set in Chicago and Missouri.',
                                                  'Jason Bateman and Laura Linney play the lead roles on the show, which is set in Missouri.',
                                                  'Laura Linney, Jason Bateman, and Julia Garner star in the show.']})
        dataset = rg.stow(dataset=dataset, cached_ops={rg.Spacy(): [['text']]})
        ef = rg.EntityFrequency([('PERSON', [3, 2, 1]), ('GPE', [1, 2, 3])])
        new_dataset, slices, slice_membership = ef(dataset, keys=['text'], store_compressed=False)
        self.assertEqual(
            slice_membership.tolist(),
            [
                [0, 0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0, 0]
            ]
        )


if __name__ == '__main__':
    unittest.main()
