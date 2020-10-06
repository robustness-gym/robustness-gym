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
        assert isinstance(pipeline, Pipeline), \
            "`pipeline` must be an nlpaug Pipeline object. " \
            "Please use \nfrom nlpaug.flow import Sequential\nrg.NlpAug(pipeline=Sequential(flow=[...]))."

        # Superclass call
        super(NlpAug, self).__init__(
            num_transformed=num_transformed,
            identifiers=Identifier.range(
                n=num_transformed,
                _name=self.__class__.__name__,
                pipeline=[Identifier(_name=augmenter.name,
                                     src=augmenter.aug_src if hasattr(augmenter, 'aug_src') else None,
                                     action=augmenter.action,
                                     method=augmenter.method)
                          for augmenter in pipeline]
            ) if not identifiers else identifiers,
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
        augmented_texts = self.pipeline.augment(
            data=column_batch,
            n=self.num_transformed,
        )

        # Transpose the list of lists from [4 x 32] to [32 x 4] and return
        return list(map(list, zip(*augmented_texts)))
