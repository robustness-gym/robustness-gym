from typing import List, Dict, Tuple

import numpy as np

from robustness_gym.identifier import Identifier
from robustness_gym.slicemakers.augmentation import Augmentation
from robustness_gym.slicemakers.augmentations.backtranslation.fairseq import load_models, batch_backtranslation


class FairseqBacktranslation(Augmentation):

    def __init__(self,
                 n_src2tgt: int = 1,
                 n_tgt2src: int = 1,
                 langs: str = 'en2de',
                 torchhub_dir: str = None,
                 device: str = 'cuda',
                 src2tgt_topk: int = 1000,
                 src2tgt_temp: float = 1.0,
                 tgt2src_topk: int = 1000,
                 tgt2src_temp: float = 1.0,
                 ):
        super(FairseqBacktranslation, self).__init__(
            identifiers=[
                Identifier(
                    name=f"{self.__class__.__name__}-{i + 1}",
                    langs=langs,
                    src2tgt_topk=src2tgt_topk,
                    src2tgt_temp=src2tgt_temp,
                    tgt2src_topk=tgt2src_topk,
                    tgt2src_temp=tgt2src_temp,
                )
                for i in range(n_src2tgt * n_tgt2src)
            ]
        )

        # Set the parameters
        self.n_src2tgt = n_src2tgt
        self.n_tgt2src = n_tgt2src
        self.src2tgt_topk = src2tgt_topk
        self.src2tgt_temp = src2tgt_temp
        self.tgt2src_topk = tgt2src_topk
        self.tgt2src_temp = tgt2src_temp

        self.num_aug = n_src2tgt * n_tgt2src

        # Setup the backtranslation model
        self.src2tgt, self.tgt2src = load_models(
            langs=langs,
            torchhub_dir=torchhub_dir,
            device=device
        )

    def apply(self,
              skeleton_batches: List[Dict[str, List]],
              slice_membership: np.ndarray,
              batch: Dict[str, List],
              keys: List[str],
              *args,
              **kwargs) -> Tuple[List[Dict[str, List]], np.ndarray]:

        for key in keys:
            # Iterate over key for all examples in the batch
            try:
                # Backtranslate
                augmented_texts = batch_backtranslation(batch[key],
                                                        src2tgt=self.src2tgt,
                                                        tgt2src=self.tgt2src,
                                                        n_src2tgt=self.n_src2tgt,
                                                        n_tgt2src=self.n_tgt2src,
                                                        src2tgt_topk=self.src2tgt_topk,
                                                        src2tgt_temp=self.src2tgt_temp,
                                                        tgt2src_topk=self.tgt2src_topk,
                                                        tgt2src_temp=self.tgt2src_temp)

                # Store the augmented text in the skeleton batches
                for i in range(slice_membership.shape[0]):
                    for j, augmented_text in enumerate(augmented_texts):
                        skeleton_batches[j][key][i] = augmented_text
            except:
                # Unable to augment: set all slice membership labels to zero
                slice_membership[:, :] = 0

        return skeleton_batches, slice_membership
