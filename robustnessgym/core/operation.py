"""Implementation of the Operation abstract base class."""
from __future__ import annotations

import pathlib
from abc import ABC
from collections import defaultdict
from functools import partial
from typing import Callable, Dict, List, Optional, Union

from meerkat import AbstractColumn
from meerkat.mixins.identifier import IdentifierMixin
from meerkat.tools.identifier import Identifier

from robustnessgym.core.slice import SliceDataPanel as DataPanel


def tuple_to_dict(keys: List[str]):
    def _tuple_to_dict(fn: callable):
        def _wrapper(*args, **kwargs):
            # Assume that if the output isn't a tuple,
            # it can be converted to a tuple of length 1
            output = fn(*args, **kwargs)
            if not isinstance(output, tuple):
                output = (output,)
            return dict(zip(keys, output))

        return _wrapper

    return _tuple_to_dict


def stow(
    dp: DataPanel,
    ops: Dict[Operation, List[List[str]]],
):
    """Apply Operations in sequence.

    Args:
        dp (DataPanel): DataPanel
        ops (Dict[Operation, List[List[str]]]):

    Returns:
        Updated DataPanel.
    """

    # Remove Operations whose outputs are already in the DataPanel
    for op, list_of_columns in list(ops.items()):
        indices_to_remove = []
        for i, columns in enumerate(list(list_of_columns)):
            if op.exists(dp):
                # Remove the columns at index i
                indices_to_remove.append(i)

        # Remove input columns for which the Operation was previously executed
        for index in sorted(indices_to_remove, reverse=True):
            columns = ops[op].pop(index)
            print(f"skipped: {op.identifier} -> {columns}", flush=True)

        # Remove the op entirely if list_of_columns is now empty
        if not ops[op]:
            ops.pop(op)

    # Run the remaining Operations
    for op, list_of_columns in ops.items():
        for columns in list_of_columns:
            dp = op(dp, columns=columns)

    return dp


def lookup(
    dp: DataPanel,
    op: Union[type, Operation],
    columns: List[str],
    output_name: str = None,
) -> AbstractColumn:
    """Retrieve the outputs of an Operation from a DataPanel.

    Args:
        dp (DataPanel): DataPanel
        op (Union[type, Operation]): subclass of Operation, or Operation object
        columns (List[str]): list of input columns that Operation was applied to
        output_name (Optional[str]): for an Operation with `num_outputs` > 1,
            the name of the output column to lookup

    Returns:
        Output columns of the Operation from the DataPanel.
    """
    # Operation identifier that should be retrieved
    if isinstance(op, Operation):
        op_name = str(op.identifier.name)
    else:
        op_name = str(Identifier(op.__name__))

    # Identifiers for all columns in the DataPanel, grouped without input columns
    # for Operation identifiers.
    column_identifiers = defaultdict(list)
    for col in dp.columns:
        identifier = Identifier.parse(col)
        column_identifiers[identifier.without("columns")].append(identifier)

    # Search for the column group that best matches the Operation identifier
    best_match, best_distance = None, 100000000
    for identifier in column_identifiers:
        # The prefix to match
        prefix = str(identifier)

        # Pick the key that best matches the cls name or instance identifier
        if (
            prefix.startswith(op_name)
            and len(
                prefix.replace(op_name, "").replace(
                    "" if output_name is None else output_name, ""
                )
            )
            < best_distance
        ):
            best_match = identifier
            best_distance = len(
                prefix.replace(op_name, "").replace(
                    "" if output_name is None else output_name, ""
                )
            )

    # Get the best matched column group
    identifier = best_match

    if identifier is None:
        raise AttributeError("Lookup failed.")

    return dp[str(identifier(columns=columns))]


class Operation(ABC, IdentifierMixin):
    """Abstract base class for operations in Robustness Gym."""

    # Path to a log directory
    logdir: pathlib.Path = pathlib.Path.home() / "robustnessgym/operations/"

    # Create a directory
    logdir.mkdir(parents=True, exist_ok=True)

    def __init__(
        self,
        identifier: Identifier = None,
        output_names: List[str] = None,
        process_batch_fn: Callable = None,
        prepare_batch_fn: Callable = None,
        **kwargs,
    ):
        super(Operation, self).__init__(
            identifier=identifier
            if identifier
            else Identifier(_name=self.__class__.__name__, **kwargs),
        )

        self._output_names = output_names

        if process_batch_fn:
            self.process_batch = process_batch_fn

        if prepare_batch_fn:
            self.prepare_batch = prepare_batch_fn

    def __repr__(self):
        return str(self.identifier)

    @property
    def num_outputs(self) -> int:
        """Number of output columns created by the Operation."""
        return len(self.output_names) if self.output_names else 1

    @property
    def output_names(self) -> Optional[List[str]]:
        """Name of output columns created by the Operation."""
        return self._output_names

    @property
    def output_identifiers(self) -> List[Identifier]:
        if self.output_names:
            return [self.identifier(output=name) for name in self.output_names]
        return [self.identifier]

    @property
    def output_columns(self) -> List[str]:
        return [str(identifier) for identifier in self.output_identifiers]

    @property
    def identifier(self) -> Identifier:
        """Name of the Operation."""
        return self._identifier

    @classmethod
    def exists(cls, dp: DataPanel) -> bool:
        """Check if the outputs of the Operation are in `dp`.

        Args:
            dp: DataPanel

        Returns:
            bool: True if `dp` contains a column produced by `Operation`,
                False otherwise
        """
        # TODO: update this to use `Operation.outputs`
        return any([key.startswith(cls.__name__) for key in dp.keys()])

    def prepare_batch(
        self,
        dp: DataPanel,
        columns: List[str],
        *args,
        **kwargs,
    ) -> None:
        """Preparation applied to the DataPanel `dp`.

        This is provided as a convenience function that can be called by
        `self.prepare`.

        Args:
            dp (DataPanel): DataPanel
            columns (list): list of columns
            *args: optional positional arguments
            **kwargs: optional keyword arguments
        """
        raise NotImplementedError("Implement `prepare_batch`.")

    def prepare(
        self,
        dp: DataPanel,
        columns: List[str],
        batch_size: int = 32,
        *args,
        **kwargs,
    ) -> None:
        """Preparation that is applied before the Operation is applied.

        Many Operations require a full pass over the DataPanel to precompute some
        variables before the core operation can actually be applied e.g. to create a
        Bag-of-Words representation, constructing a vocabulary to keep only
        tokens that are frequently seen across the DataPanel.

        Args:
            dp (DataPanel): DataPanel
            columns (list): list of columns
            batch_size (int): batch size for `dp.map(...)`
            *args: optional positional arguments
            **kwargs: optional keyword arguments
        """

        try:
            dp.map(
                function=partial(self.prepare_batch, columns=columns, *args, **kwargs),
                input_columns=columns,
                is_batched_fn=True,
                batch_size=batch_size,
                *args,
                **kwargs,
            )
        except NotImplementedError:
            return

    def process_batch(
        self,
        dp: DataPanel,
        columns: List[str],
        **kwargs,
    ) -> tuple:
        """The core functionality of the Operation.

        This is provided as a convenience function that can be called by
        `self.process`.

        Args:
            dp (DataPanel): DataPanel
            columns (list): list of columns
            **kwargs: optional keyword arguments

        Returns:
            Tuple of outputs, where each output is a a sequence of values. The expected
                order of the outputs is the same as the order of identifiers in
                `self.outputs`.
        """
        return NotImplemented

    def process(
        self,
        dp: DataPanel,
        columns: List[str],
        batch_size: int = 32,
        *args,
        **kwargs,
    ) -> DataPanel:
        """Apply the Operation to a DataPanel.

        Args:
            dp (DataPanel): DataPanel
            columns (list): list of columns
            batch_size (int): batch size for `dp.update(...)`
            *args: optional positional arguments
            **kwargs: optional keyword arguments
        """

        return dp.update(
            tuple_to_dict(
                keys=[str(ident(columns=columns)) for ident in self.output_identifiers]
            )(partial(self.process_batch, columns=columns, *args, **kwargs)),
            batch_size=batch_size,
            is_batched_fn=True,
            *args,
            **kwargs,
        )

    def __call__(
        self,
        dp: DataPanel,
        columns: List[str],
        batch_size: int = 32,
        **kwargs,
    ) -> DataPanel:
        """Apply the Operation to a DataPanel.

        Args:
            dp (DataPanel): DataPanel
            columns (list): list of columns
            batch_size (int):
            **kwargs: optional keyword arguments

        Returns:
            An updated DataPanel, with additional output columns produced by
              the Operation.
        """

        if isinstance(dp, DataPanel):
            assert len(set(columns) - set(dp.column_names)) == 0, (
                f"All `columns` ({columns}) must be present and visible in `dp` ("
                f"{list(dp.column_names)})."
            )

            if self.exists(dp):
                return dp

            # Prepare to apply the Operation to the DataPanel
            self.prepare(
                dp=dp,
                columns=columns,
                batch_size=batch_size,
                **kwargs,
            )

            # Apply the Operation to the DataPanel
            dp = self.process(
                dp=dp,
                columns=columns,
                batch_size=batch_size,
                **kwargs,
            )

            return dp

        else:
            return self(
                dp=DataPanel(dp),
                columns=columns,
                batch_size=batch_size,
                **kwargs,
            )
