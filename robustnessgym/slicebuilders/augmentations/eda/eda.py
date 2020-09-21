from typing import List, Dict, Tuple

import numpy as np

from robustnessgym.identifier import Identifier
from robustnessgym.slicebuilders.augmentation import Augmentation
from robustnessgym.slicebuilders.augmentations.eda._eda import eda


class EasyDataAugmentation(Augmentation):

    def __init__(self,
                 num_aug=1,
                 alpha_sr=0.1,
                 alpha_ri=0.1,
                 alpha_rs=0.1,
                 p_rd=0.1):

        super(EasyDataAugmentation, self).__init__(
            identifiers=[
                Identifier(
                    _name=f"{self.__class__.__name__}-{i + 1}",
                    alpha_sr=alpha_sr,
                    alpha_ri=alpha_ri,
                    alpha_rs=alpha_rs,
                    p_rd=p_rd,
                )
                for i in range(num_aug)
            ]
        )

        # Set the parameters
        self.num_aug = num_aug
        self.alpha_sr = alpha_sr
        self.alpha_ri = alpha_ri
        self.alpha_rs = alpha_rs
        self.p_rd = p_rd

        # Download wordnet
        self._download_wordnet()

    def _download_wordnet(self):
        try:
            if not (self.logdir / 'wordnet').exists():
                import nltk
                nltk.download('wordnet', download_dir=str(self.logdir / 'wordnet'), quiet=False)
        except ImportError:
            print("Need nltk.")

    def apply(self,
              skeleton_batches: List[Dict[str, List]],
              slice_membership: np.ndarray,
              batch: Dict[str, List],
              columns: List[str],
              *args,
              **kwargs) -> Tuple[List[Dict[str, List]], np.ndarray]:

        for key in columns:
            # Iterate over key for all examples in the batch
            for i, text in enumerate(batch[key]):
                try:
                    # EDA returns a list of augmented text, including the original text at the last position
                    augmented_texts = eda(text,
                                          alpha_sr=self.alpha_sr,
                                          alpha_ri=self.alpha_ri,
                                          alpha_rs=self.alpha_rs,
                                          p_rd=self.p_rd,
                                          num_aug=self.num_aug)[:-1]

                    # Store the augmented text in the augmented batches
                    for j, augmented_text in enumerate(augmented_texts):
                        skeleton_batches[j][key][i] = augmented_text
                except:
                    # Unable to augment the example: set its slice membership to zero
                    slice_membership[i, :] = 0

        return skeleton_batches, slice_membership
