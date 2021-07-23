"""Transformations using nlpaug."""
from typing import List

from meerkat.tools.lazy_loader import LazyLoader

from robustnessgym.core.identifier import Identifier
from robustnessgym.slicebuilders.transformation import SingleColumnTransformation

nlpaug_flow = LazyLoader("nlpaug.flow", error="Please `pip install nlpaug`.")


class NlpAugTransformation(SingleColumnTransformation):
    """Class for building transformations using nlpaug."""

    def __init__(
        self,
        pipeline: "nlpaug_flow.Pipeline",
        num_transformed: int = 1,
        identifiers: List[Identifier] = None,
        *args,
        **kwargs
    ):
        assert isinstance(pipeline, nlpaug_flow.Pipeline), (
            "`pipeline` must be an nlpaug Pipeline object. Please use \n"
            "from nlpaug.flow import Sequential\n"
            "rg.NlpAugTransformation(pipeline=Sequential(flow=[...]))."
        )

        super(NlpAugTransformation, self).__init__(
            num_transformed=num_transformed,
            identifiers=Identifier.range(
                n=num_transformed,
                _name=self.__class__.__name__,
                pipeline=[
                    Identifier(
                        _name=augmenter.name,
                        src=augmenter.aug_src
                        if hasattr(augmenter, "aug_src")
                        else None,
                        action=augmenter.action,
                        method=augmenter.method,
                    )
                    for augmenter in pipeline
                ],
            )
            if not identifiers
            else identifiers,
            *args,
            **kwargs
        )

        # Set the pipeline
        self.pipeline = pipeline

    def single_column_apply(self, column_batch: List[str]) -> List[List[str]]:
        # Apply the nlpaug pipeline
        augmented_texts = self.pipeline.augment(
            data=column_batch,
            n=self.num_transformed,
        )

        if self.num_transformed == 1:
            augmented_texts = [augmented_texts]

        # Transpose the list of lists from [4 x 32] to [32 x 4] and return
        return list(map(list, zip(*augmented_texts)))
