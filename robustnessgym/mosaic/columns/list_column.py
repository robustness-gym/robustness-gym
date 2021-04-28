from __future__ import annotations

import abc
import logging
from collections import defaultdict
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
from tqdm.auto import tqdm
from yaml.representer import Representer

from robustnessgym.core.tools import convert_to_batch_column_fn
from robustnessgym.mosaic.columns.abstract import AbstractColumn

Representer.add_representer(abc.ABCMeta, Representer.represent_name)


logger = logging.getLogger(__name__)


# Q. how to handle collate and materialize here? Always materialized but only sometimes
# may want to collate (because collate=True should return a batch-style object, while
# collate=False should return a Column style object).


class ListColumn(AbstractColumn):
    def __init__(
        self,
        data: Sequence = None,
        *args,
        **kwargs,
    ):
        self._data = data
        super(ListColumn, self).__init__(num_rows=len(self), *args, **kwargs)

    @classmethod
    def from_list(cls, data: Sequence, *args, **kwargs):
        return cls(data=data, *args, **kwargs)

    def __getitem__(self, index):
        if self.visible_rows is not None:
            # Remap the index if only some rows are visible
            index = self._remap_index(index)

        # `index` should return a single element
        if isinstance(index, int) or isinstance(index, np.int):
            return self.data[int(index)]

        # indices that return batches
        if isinstance(index, slice):
            # int or slice index => standard list slicing
            data = self.data[index]
        elif (isinstance(index, tuple) or isinstance(index, list)) and len(index):
            data = [self.data[i] for i in index]
        elif isinstance(index, np.ndarray) and len(index.shape) == 1:
            data = [self.data[int(i)] for i in index]
        else:
            raise TypeError("Invalid argument type: {}".format(type(index)))

        # TODO(karan): do we need collate in ListColumn
        # if self._materialize:  # self._collate
        #     # return a batch
        #     return self.collate([element for element in data])
        # else:
        #     # if not materializing, return a new ListColumn
        #     return self.from_list(data)
        return self.from_list(data)

    def batch(
        self,
        batch_size: int = 32,
        drop_last_batch: bool = False,
        # collate: bool = True,
        *args,
        **kwargs,
    ):
        # TODO(karan): do we need collate in ListColumn
        # if self._materialize:
        #     return torch.utils.data.DataLoader(
        #         self,
        #         batch_size=batch_size,
        #         collate_fn=self.collate if collate else identity_collate,
        #         drop_last=drop_last_batch,
        #         *args,
        #         **kwargs,
        #     )
        # else:
        #     return super(ListColumn, self).batch(
        #         batch_size=batch_size, drop_last_batch=drop_last_batch
        #     )
        return super(ListColumn, self).batch(
            batch_size=batch_size, drop_last_batch=drop_last_batch
        )

    def map(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        drop_last_batch: bool = False,
        num_proc: Optional[int] = None,
        materialize: bool = None,
        **kwargs,
    ) -> Optional[Union[Dict, List]]:
        """Apply a map over the dataset."""
        # Check if need to materialize:
        # TODO(karan): figure out if we need materialize=False

        # Just return if the function is None
        if function is None:
            logger.info("`function` None, returning None.")
            return None

        # Ensure that num_proc is not None
        if num_proc is None:
            num_proc = 0

        # Return if `self` has no examples
        if not len(self):
            logger.info("Dataset empty, returning None.")
            return None

        if not batched:
            # Convert to a batch function
            function = convert_to_batch_column_fn(function, with_indices=with_indices)
            # TODO: Transfer this fix to other classes
            batched = True
            logger.info(f"Converting `function` {function} to a batched function.")

        # # Get some information about the function
        # TODO: discuss whether this is actually required vs. doing it on first pass in
        # loop
        function_properties = self._inspect_function(
            function,
            with_indices,
            batched=batched,
        )

        # Run the map
        logger.info("Running `map`, the dataset will be left unchanged.")
        outputs = defaultdict(list) if function_properties.dict_output else []
        for i, batch in tqdm(
            enumerate(
                self.batch(
                    batch_size=batch_size,
                    drop_last_batch=drop_last_batch,
                    # collate=batched,
                )
            ),
            total=(len(self) // batch_size)
            + int(not drop_last_batch and len(self) % batch_size != 0),
        ):

            # Run `function` on the batch
            output = (
                function(
                    batch,
                    range(i * batch_size, min(len(self), (i + 1) * batch_size)),
                )
                if with_indices
                else function(batch)
            )

            # Append the output
            if output is not None:
                if isinstance(output, Mapping):
                    for k in output.keys():
                        outputs[k].extend(output[k])
                else:
                    outputs.extend(output)

        if not len(outputs):
            return None
        elif isinstance(outputs, dict):
            # turns the defaultdict into dict
            return dict(outputs)
        return outputs
