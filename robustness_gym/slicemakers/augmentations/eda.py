from robustness_gym.slicemaker import *
from robustness_gym.slicemakers.augmentations._eda import eda


class EasyDataAugmentation(Augmentation):

    def __init__(self,
                 num_aug=1,
                 alpha_sr=0.1,
                 alpha_ri=0.1,
                 alpha_rs=0.1,
                 p_rd=0.1):
        super(EasyDataAugmentation, self).__init__(
            num_slices=num_aug,
        )

        # Set the parameters
        self.num_aug = num_aug
        self.alpha_sr = alpha_sr
        self.alpha_ri = alpha_ri
        self.alpha_rs = alpha_rs
        self.p_rd = p_rd

    def alias(self):
        return self.__class__.__name__

    def process_batch(self,
                      batch: Dict[str, List],
                      keys: List[str]) -> Tuple[Dict[str, List], List[Dict[str, List]], Optional[np.ndarray]]:

        # Uncache the batch to construct the skeleton for augmented batches
        augmented_batches = [Dataset.uncached_batch(batch) for _ in range(self.num_aug)]

        # Set the index for the augmented batches
        for j, augmented_batch in enumerate(augmented_batches):
            augmented_batch['index'] = [f'{idx}-{self.alias()}-{j}' for idx in augmented_batch['index']]

        slice_labels = np.ones((len(batch['index']), self.num_aug))

        for key in keys:
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
                        augmented_batches[j][key][i] = augmented_text
                except:
                    # Unable to augment the example: set its slice label to zero
                    slice_labels[i, :] = 0

        # Remove augmented examples where slice_labels[i, :] = 0
        augmented_batches = [self.slice_batch_with_slice_labels(aug_batch, slice_labels[:, j:j + 1])[0]
                             for j, aug_batch in enumerate(augmented_batches)]

        # Update the batch to cache the augmented examples
        batch = self.store_augmentations(batch, augmented_batches, self.alias())

        # EDA is assumed to be label-preserving, so we don't change anything else
        return batch, augmented_batches, slice_labels
