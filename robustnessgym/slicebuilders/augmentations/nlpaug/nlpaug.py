from typing import List

import cytoolz as tz
from nlpaug.flow import Pipeline

from robustnessgym.identifier import Identifier
from robustnessgym.slicebuilders.augmentation import SingleColumnAugmentation


class NlpAug(SingleColumnAugmentation):

    def __init__(self,
                 pipeline: Pipeline,
                 num_transformed: int = 1,
                 identifiers: List[Identifier] = None,
                 *args,
                 **kwargs):
        # Superclass call
        super(NlpAug, self).__init__(
            num_transformed=num_transformed,
            identifiers=[
                Identifier(
                    name=f"{self.__class__.__name__}-{i + 1}",
                    pipeline=[Identifier(name=augmenter.name,
                                         src=augmenter.aug_src,
                                         action=augmenter.action,
                                         method=augmenter.method)
                              for augmenter in pipeline],
                )
                for i in range(num_transformed)
            ] if not identifiers else identifiers,
            *args,
            **kwargs
        )

        # Set the pipeline
        self._pipeline = pipeline

    @property
    def pipeline(self):
        return self._pipeline

    def single_column_apply(self,
                            column_batch: List[str]) -> List[List[str]]:
        # Apply the nlpaug pipeline
        augmented_texts = self.pipeline.augments(
            data=column_batch,
            n=self.num_transformed,
        )

        # Partition to reshape and return
        return list(tz.partition_all(self.num_transformed, augmented_texts))
