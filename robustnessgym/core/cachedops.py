from typing import Callable, List

from robustnessgym.core.slice import SliceDataPanel as DataPanel
from robustnessgym.core.decorators import singlecolumn
from robustnessgym.core.identifier import Identifier
from robustnessgym.core.operation import Operation


class NamedColumn(Operation):
    """Class to create NamedColumns."""

    def __init__(
        self,
        apply_fn: Callable = None,
        identifier: Identifier = None,
        *args,
        **kwargs,
    ):

        super(NamedColumn, self).__init__(
            apply_fn=apply_fn,
            identifiers=[identifier] if identifier else None,
            num_outputs=1,
            *args,
            **kwargs,
        )


class SingleColumnCachedOperation(NamedColumn):
    def __call__(
        self, batch_or_dataset: DataPanel, columns: List[str], batch_size: int = 32
    ) -> DataPanel:
        """Apply independently to each column.

        Args:
            batch_or_dataset:
            columns:

        Returns:
        """
        # Iterate over the columns and apply
        for column in columns:
            batch_or_dataset = super(SingleColumnCachedOperation, self).__call__(
                batch_or_dataset=batch_or_dataset,
                columns=[column],
                batch_size=batch_size,
            )

        return batch_or_dataset

    @singlecolumn
    def apply(self, dp: DataPanel, columns: List[str], *args, **kwargs) -> List:
        return self.single_column_apply(dp[columns[0]])

    def single_column_apply(self, column_batch: List, **kwargs) -> List:
        raise NotImplementedError("Must implement single_column_apply.")


