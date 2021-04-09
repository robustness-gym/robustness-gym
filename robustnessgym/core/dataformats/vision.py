from __future__ import annotations

import copy
import gzip
import logging
import os
import pickle
import tempfile
import uuid
from collections import defaultdict
from types import SimpleNamespace
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Union

import cytoolz as tz
import datasets
import numpy as np
import pyarrow as pa
import torch
from datasets import DatasetInfo, Features, NamedSplit
from joblib import Parallel, delayed
from torch.utils.data._utils.collate import default_collate
from tqdm.auto import tqdm

from robustnessgym.core.dataformats.abstract import AbstractDataset
from robustnessgym.core.tools import convert_to_batch_fn

logger = logging.getLogger(__name__)

Example = Dict
Batch = Dict[str, List]


class RGImage:
    """This class acts as an interface to allow the user to manipulate the
    images without actually loading them into memory."""

    def __init__(self, filepath: str, transform: callable = None):
        self.filepath = filepath
        self.name = os.path.split(filepath)[-1]

        # Cache the transforms applied on the image when VisionDataset.update
        # gets called
        self.transform = transform

    def display(self):
        pass

    def load(self):
        import torchvision.datasets.folder as folder

        image = torch.from_numpy(np.array(folder.default_loader(self.filepath)))

        if self.transform is not None:
            image = self.transform(image)

        return image

    def __getitem__(self, idx):
        image = self.load()
        return image[idx]

    def __str__(self):
        return self.filepath

    def __repr__(self):
        return "Image(%s)" % self.name

    def __eq__(self, other):
        filepath_eq = self.filepath == other.filepath
        transform_eq = self.transform == other.transform
        return filepath_eq and transform_eq


# TODO(sabri): discuss merging RGImageRow and RGImageBatch into one class
class RGImageRow(dict):
    """Instances of this class are returned from
    `VisionDataset.__getitem__(index)` when a single row is indexed (i.e.
    `index` is an `int` or `np.int`).

    These instances should behave just as regular dictionaries do and
    should be largely indistinguishable to the end user with the only
    difference being that image columns are loaded lazily.
    """

    @classmethod
    def from_dict(
        cls,
        row: Dict[str, List],
        img_columns: Union[str, List[str]],
    ):
        row = cls(**row)
        if isinstance(img_columns, str):
            img_columns = [img_columns]
        row._img_columns = img_columns

        return row

    def __getitem__(self, key: str):
        val = super(RGImageRow, self).__getitem__(key)
        if key in self._img_columns:
            return val.load()

        return super(RGImageRow, self).__getitem__(key)


class RGImageBatch(dict):
    """Instances of this class are returned from
    `VisionDataset.__getitem__(index)` when multiple rows are indexed (i.e.
    `index` is a `slice`, `np.ndarray`, etc.).

    To the end user, these instances should be largely indistinguishable
    from a batch represented by dictionary (like those returned by
    `InMemoryDataset`) with the main difference being being that image
    columns are loaded lazily and collated (i.e. stacked) into one
    tensor only when those columns are accessed.
    """

    @classmethod
    def from_dict(
        cls,
        batch: Dict[str, List],
        img_columns: Union[str, List[str]],
        collate_fn: callable,
    ):
        batch = cls(**batch)
        batch.collate_fn = collate_fn
        if isinstance(img_columns, str):
            img_columns = [img_columns]
        batch._img_columns = img_columns

        return batch

    def __getitem__(self, key: str):
        val = super(RGImageBatch, self).__getitem__(key)

        # in `VisionDataset.map`, we use a torch DataLoader to load the images and then
        # set the value in the batch to the loaded tensor. In this case
        # batch[img_key] will already be a collated tensor, hence the check for
        # `torch.is_tensor`
        if key in self._img_columns and not torch.is_tensor(val):
            return self.collate_fn([img.load() for img in val])

        return super(RGImageBatch, self).__getitem__(key)


def save_image(image, filename):
    """Save 'image' to file 'filename' and return an RGImage object."""
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    from PIL import Image as im

    image = im.fromarray(image.astype(np.uint8))
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(filename)

    return RGImage(filename)


class VisionDataset(AbstractDataset):
    """Class for vision datasets that are to be stored in memory."""

    def __init__(
        self,
        *args,
        column_names: List[str] = None,
        info: DatasetInfo = None,
        split: Optional[NamedSplit] = None,
        img_columns: Union[str, List[str]] = ["image_file"],
        transform: Union[callable, Mapping[str, callable]] = None,
    ):

        # Data is a dictionary of lists
        self._data = {}

        if isinstance(img_columns, str):
            img_columns = [img_columns]
        self._img_columns = img_columns

        self._img_key_to_transform = defaultdict(lambda: None)
        if isinstance(transform, Mapping):
            if (set(transform.keys()) - self._img_columns) != 0:
                raise ValueError(
                    "Mapping passed to `transforms` includes keys not in `img_columns`."
                )
            self._img_key_to_transform.update(transform)
        elif transform is not None:
            # if single `callable` is passed, use same transform for all image columns
            self._img_key_to_transform.update({k: transform for k in self._img_columns})

        self.collate_fn = default_collate

        # Internal state variables for the update function
        self._updating_images = False
        self._adding_images = False
        self._callstack = []

        # Single argument
        if len(args) == 1:
            assert column_names is None, "Don't pass in column_names."
            # The data is passed in
            data = args[0]

            # TODO: Discuss whether this is a worthy refactor
            # This replaces the commented elif block below
            if isinstance(data, list) and len(data):
                # Transpose the list of dicts to a dict of lists i.e. a batch
                data = tz.merge_with(list, *data)

            # `data` is a dictionary
            if isinstance(data, dict) and len(data):
                # Assert all columns are the same length
                self._assert_columns_all_equal_length(data)
                mask = [key in data for key in self._img_columns]
                if not all(mask):
                    idx = mask.index(False)
                    raise KeyError(
                        "Key with paths to images not found: %s"
                        % self._img_columns[idx]
                    )
                for key in self._img_columns:
                    self._paths_to_Images(data, key)
                self._data = data

            # `data` is a list
            # elif isinstance(data, list) and len(data):
            #     # Transpose the list of dicts to a dict of lists i.e. a batch
            #     data = tz.merge_with(list, *data)
            #     # Assert all columns are the same length
            #     self._assert_columns_all_equal_length(data)
            #     if filepath_key not in data:
            #         raise KeyError(
            #             "Key with paths to images not found: %s" % filepath_key
            #         )
            #     self._paths_to_Images(data, filepath_key)
            #     self._data = data

            # `data` is a datasets.Dataset
            elif isinstance(data, datasets.Dataset):
                self._data = data[:]
                info, split = data.info, data.split

        # No argument
        elif len(args) == 0:

            # Use column_names to setup the data dictionary
            if column_names:
                self._data = {k: [] for k in column_names}

        else:
            raise NotImplementedError(
                "Currently only one table is supported when creating a VisionDataSet"
            )

        # Setup the DatasetInfo
        info = info.copy() if info is not None else DatasetInfo()
        AbstractDataset.__init__(self, info=info, split=split)

        # Create attributes for all columns and visible columns
        self.all_columns = list(self._data.keys())
        self.visible_columns = None

        # Create attributes for visible rows
        self.visible_rows = None

        # Initialization
        self._initialize_state()

        logger.info(
            f"Created `VisionDataset` with {len(self)} rows and "
            f"{len(self.column_names)} columns."
        )

    def _paths_to_Images(self, data, key):
        """Convert a list of paths to images data[key] into a list of RGImage
        instances."""
        if isinstance(data[key][0], RGImage):
            return  # Can happen when we're copying a dataset
        data[key] = [
            RGImage(i, transform=self._img_key_to_transform[key]) for i in data[key]
        ]

    def _set_features(self):
        """Set the features of the dataset."""
        with self.format():
            d = {
                k: [""] if (k in self._img_columns or torch.torch.is_tensor(v)) else [v]
                for k, v in self[0].items()
            }
            self.info.features = Features.from_arrow_schema(
                pa.Table.from_pydict(
                    d,
                ).schema
            )

    def _materialize(self):
        # Materialize data, instead of using a reference to an ancestor Dataset
        self._data = {k: self[k] for k in self._data}

        # Reset visible_rows
        self.set_visible_rows(None)

    def add_column(self, column: str, values: List, overwrite=False) -> None:
        """Add a column to the dataset."""

        assert (
            column not in self.all_columns
        ) or overwrite, (
            f"Column `{column}` already exists, set `overwrite=True` to overwrite."
        )
        assert len(values) == len(self), (
            f"`add_column` failed. "
            f"Values length {len(values)} != dataset length {len(self)}."
        )

        if self.visible_rows is not None:
            # Materialize the data
            self._materialize()

        # Add the column
        self._data[column] = list(values)
        self.all_columns.append(column)
        self.visible_columns.append(column)

        # Set features
        self._set_features()

        logger.info(f"Added column `{column}` with length `{len(values)}`.")

    def remove_column(self, column: str) -> None:
        """Remove a column from the dataset."""
        assert column in self.all_columns, f"Column `{column}` does not exist."

        # Remove the column
        del self._data[column]
        self.all_columns = [col for col in self.all_columns if col != column]
        self.visible_columns = [col for col in self.visible_columns if col != column]

        # Set features
        self._set_features()

        logger.info(f"Removed column `{column}`.")

    def select_columns(self, columns: List[str]) -> Batch:
        """Select a subset of columns."""
        for col in columns:
            assert col in self._data
        return tz.keyfilter(lambda k: k in columns, self._data)

    def _append_to_empty_dataset(self, example_or_batch: Union[Example, Batch]) -> None:
        """Append a batch of data to the dataset when it's empty."""
        # Convert to batch
        batch = self._example_or_batch_to_batch(example_or_batch)

        # TODO(karan): what other data properties need to be in sync here
        self.all_columns = list(batch.keys())
        self.visible_columns = list(batch.keys())

        # Dataset is empty: create the columns and append the batch
        self._data = {k: [] for k in self.column_names}
        for k in self.column_names:
            self._data[k].extend(batch[k])

    def append(
        self,
        example_or_batch: Union[Example, Batch],
    ) -> None:
        """Append a batch of data to the dataset.

        `batch` must have the same columns as the dataset (regardless of
        what columns are visible).
        """
        if not self.column_names:
            return self._append_to_empty_dataset(example_or_batch)

        # Check that example_or_batch has the same format as the dataset
        # TODO(karan): require matching on nested features?
        columns = list(example_or_batch.keys())
        assert set(columns) == set(
            self.column_names
        ), f"Mismatched columns\nbatch: {columns}\ndataset: {self.column_names}"

        # Convert to a batch
        batch = self._example_or_batch_to_batch(example_or_batch)

        # Append to the dataset
        for k in self.column_names:
            if k in self._img_columns:
                batch[k] = list(map(RGImage, batch[k]))
            self._data[k].extend(batch[k])

    def _remap_index(self, index):
        if isinstance(index, int):
            return self.visible_rows[index].item()
        elif isinstance(index, slice):
            return self.visible_rows[index].tolist()
        elif isinstance(index, str):
            return index
        elif (isinstance(index, tuple) or isinstance(index, list)) and len(index):
            return self.visible_rows[index].tolist()
        elif isinstance(index, np.ndarray) and len(index.shape) == 1:
            return self.visible_rows[index].tolist()
        else:
            raise TypeError("Invalid argument type: {}".format(type(index)))

    def __getitem__(self, index):
        if self.visible_rows is not None:
            # Remap the index if only some rows are visible
            index = self._remap_index(index)
        if isinstance(index, str):
            # str index => column selection
            if index in self.column_names:
                if self.visible_rows is not None:
                    return [self._data[index][i] for i in self.visible_rows]
                return self._data[index]
            raise AttributeError(f"Column {index} does not exist.")

        if isinstance(index, int) or isinstance(index, np.int):
            return RGImageRow.from_dict(
                {k: self._data[k][index] for k in self.visible_columns},
                img_columns=self._img_columns,
            )

        # indices that return batches
        if isinstance(index, slice):
            # int or slice index => standard list slicing
            batch = {k: self._data[k][index] for k in self.visible_columns}
        elif (isinstance(index, tuple) or isinstance(index, list)) and len(index):
            batch = {k: [self._data[k][i] for i in index] for k in self.visible_columns}
        elif isinstance(index, np.ndarray) and len(index.shape) == 1:
            batch = {
                k: [self._data[k][int(i)] for i in index] for k in self.visible_columns
            }
        else:
            raise TypeError("Invalid argument type: {}".format(type(index)))

        return RGImageBatch.from_dict(
            batch, img_columns=self._img_columns, collate_fn=self.collate_fn
        )

    def _inspect_update_function(
        self,
        function: Callable,
        with_indices: bool = False,
        batched: bool = False,
    ) -> SimpleNamespace:
        """Load the images before calling _inspect_function, and check if new
        image columns are being added."""

        properties = self._inspect_function(function, with_indices, batched)

        # Check if new columns are added
        if batched:
            if with_indices:
                output = function(self[:2], range(2))
            else:
                output = function(self[:2])

        else:
            if with_indices:
                output = function(self[0], 0)
            else:
                output = function(self[0])
        new_columns = set(output.keys()).difference(set(self.all_columns))

        # Check if any of those new columns is an image column
        new_img_columns = []
        for key in new_columns:
            val = output[key]
            if isinstance(val, torch.Tensor) and len(val.shape) >= 2:
                new_img_columns.append(key)

        properties.new_image_columns = new_img_columns

        return properties

    def update(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        # input_columns: Optional[Union[str, List[str]]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        remove_columns: Optional[List[str]] = None,
        cache_dir: str = None,
        **kwargs,
    ) -> Optional[VisionDataset]:
        """Update the columns of the dataset."""
        # TODO(karan): make this fn go faster
        # most of the time is spent on the merge, speed it up further

        # Sanity check when updating the images
        self._callstack.append("update")

        # Return if the function is None
        if function is None:
            logger.info("`function` None, returning None.")
            return self

        # Return if `self` has no examples
        if not len(self):
            logger.info("Dataset empty, returning None.")
            return self

        # Get some information about the function
        function_properties = self._inspect_update_function(
            function, with_indices, batched
        )
        assert (
            function_properties.dict_output
        ), f"`function` {function} must return dict."

        if not batched:
            # Convert to a batch function
            function = convert_to_batch_fn(function, with_indices=with_indices)
            logger.info(f"Converting `function` {function} to batched function.")

        updated_columns = function_properties.existing_columns_updated
        changed_images = [key in self._img_columns for key in updated_columns]
        new_image_columns = function_properties.new_image_columns

        # Set the internal state for the map function
        self._updating_images = any(changed_images)
        self._adding_images = any(new_image_columns)
        if self._updating_images or self._adding_images:
            # Set the cache directory where the modified images will be stored
            if not cache_dir:
                cache_dir = tempfile.gettempdir()
                logger.warning(
                    "Modifying the images without setting a cache directory.\n"
                    "Consider setting it if your dataset is very large.\n"
                    "The default image cache location is: {}".format(cache_dir)
                )

            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)

            cache_dir = os.path.join(cache_dir, uuid.uuid4().hex)
            os.mkdir(cache_dir)

        # Update always returns a new dataset
        logger.info("Running update, a new dataset will be returned.")
        if self.visible_rows is not None:
            # Run .map() to get updated batches and pass them into a new dataset
            new_dataset = VisionDataset(
                self.map(
                    (
                        lambda batch, indices: self._merge_batch_and_output(
                            batch, function(batch, indices)
                        )
                    )
                    if with_indices
                    else (
                        lambda batch: self._merge_batch_and_output(
                            batch, function(batch)
                        )
                    ),
                    with_indices=with_indices,
                    batched=True,
                    batch_size=batch_size,
                    cache_dir=cache_dir,
                ),
                img_columns=self._img_columns,
            )
        else:
            if function_properties.updates_existing_column:
                # Copy the ._data dict with a reference to the actual columns
                new_dataset = self.copy()

                # Calculate the values for the updated columns using a .map()
                output = self.map(
                    (
                        lambda batch, indices:
                        # Only merge columns that get updated
                        self._merge_batch_and_output(
                            {
                                k: v
                                for k, v in batch.items()
                                if k in function_properties.existing_columns_updated
                            },
                            function(batch, indices),
                        )
                    )
                    if with_indices
                    else (
                        lambda batch:
                        # Only merge columns that get updated
                        self._merge_batch_and_output(
                            {
                                k: v
                                for k, v in batch.items()
                                if k in function_properties.existing_columns_updated
                            },
                            function(batch),
                        )
                    ),
                    with_indices=with_indices,
                    batched=True,
                    batch_size=batch_size,
                    cache_dir=cache_dir,
                    new_image_columns=new_image_columns,
                )

                # If new image columns were added, update that information
                if self._adding_images:
                    new_dataset._img_columns.extend(new_image_columns)

                # Add new columns / overwrite existing columns for the update
                for col, vals in output.items():
                    if isinstance(vals[0], torch.Tensor) and vals[
                        0
                    ].shape == torch.Size([]):
                        # Scalar tensor. Convert to Python.
                        new_vals = []
                        for val in vals:
                            new_vals.append(val.item())
                        vals = new_vals
                    new_dataset.add_column(col, vals, overwrite=True)
            else:
                # Copy the ._data dict with a reference to the actual columns
                new_dataset = self.copy()

                # Calculate the values for the new columns using a .map()
                output = new_dataset.map(
                    function=function,
                    with_indices=with_indices,
                    batched=True,
                    batch_size=batch_size,
                    cache_dir=cache_dir,
                    new_image_columns=new_image_columns,
                )

                # If new image columns were added, update that information
                if self._adding_images:
                    new_dataset._img_columns.extend(new_image_columns)

                # Add new columns for the update
                for col, vals in output.items():
                    if isinstance(vals[0], torch.Tensor) and vals[
                        0
                    ].shape == torch.Size([]):
                        # Scalar tensor. Convert to Python.
                        new_vals = []
                        for val in vals:
                            new_vals.append(val.item())
                        vals = new_vals
                    new_dataset.add_column(col, vals)

        # Remove columns
        if remove_columns:
            for col in remove_columns:
                new_dataset.remove_column(col)
            logger.info(f"Removed columns {remove_columns}.")
        # Reset the format
        # if input_columns:
        #     self.set_format(previous_format)

        # Remember to reset the internal state
        self._updating_images = False
        self._adding_images = False
        # And remove this call from the callstack
        self._callstack.pop()

        # If the new dataset is a copy we also need to reset it
        new_dataset._updating_images = False
        new_dataset._adding_images = False
        new_dataset._callstack.pop()

        return new_dataset

    def map(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        input_columns: Optional[Union[str, List[str]]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        drop_last_batch: bool = False,
        num_proc: Optional[int] = 64,
        **kwargs,
    ) -> Optional[Union[Dict, List]]:
        """Apply a map over the dataset."""

        # Just return if the function is None
        if function is None:
            logger.info("`function` None, returning None.")
            return None

        # Ensure that num_proc is not None
        if num_proc is None:
            num_proc = 64

        # Return if `self` has no examples
        if not len(self):
            logger.info("Dataset empty, returning None.")
            return None

        if isinstance(input_columns, str):
            input_columns = [input_columns]

        # Set the format
        previous_format = self.visible_columns
        if input_columns:
            self.set_format(input_columns)

        if not batched:
            # Convert to a batch function
            function = convert_to_batch_fn(function, with_indices=with_indices)
            logger.info(f"Converting `function` {function} to a batched function.")

        # Check if any of the columns is an image column
        if not input_columns:
            input_columns = self.visible_columns
        image_loaders = {}
        for key in input_columns:
            if key in self._img_columns:
                # Load the images
                images = self.to_dataloader(
                    columns=[key],
                    batch_size=batch_size,
                    shuffle=False,
                    drop_last=False,
                    num_workers=num_proc,
                    collate_fn=self.collate_fn,
                )
                images = iter(images)
                image_loaders[key] = images

        # If we are updating, prepare image savers and perform sanity checks
        if self._updating_images or self._adding_images:
            assert "update" in self._callstack, (
                "_updating_images and _adding_images can only be set by "
                "VisionDataset.update"
            )
            assert "cache_dir" in kwargs, "No cache directory specified"
            cache_dir = kwargs["cache_dir"]
        if self._adding_images:
            assert "new_image_columns" in kwargs, "New image column names not specified"
            new_image_columns = kwargs["new_image_columns"]

        # Run the map
        logger.info("Running `map`, the dataset will be left unchanged.")
        outputs = None
        for i, batch in tqdm(
            enumerate(self.batch(batch_size, drop_last_batch)),
            total=(len(self) // batch_size)
            + int(not drop_last_batch and len(self) % batch_size != 0),
        ):
            for key in image_loaders:
                batch[key] = next(image_loaders[key])

            # Run `function` on the batch
            output = (
                function(
                    batch,
                    range(i * batch_size, min(len(self), (i + 1) * batch_size)),
                )
                if with_indices
                else function(batch)
            )

            # Save the modified images
            if self._updating_images:
                for key in image_loaders:
                    images = output[key]

                    # Save the images in parallel
                    rgimages = Parallel(n_jobs=num_proc)(
                        delayed(save_image)(
                            images[idx],
                            os.path.join(
                                cache_dir,
                                "{0}{1}.png".format(key, i * batch_size + idx),
                            ),
                        )
                        for idx in range(len(images))
                    )

                    output[key] = rgimages

            if self._adding_images:
                for key in new_image_columns:
                    images = output[key]

                    # Save the images in parallel
                    rgimages = Parallel(n_jobs=num_proc)(
                        delayed(save_image)(
                            images[idx],
                            os.path.join(
                                cache_dir,
                                "{0}{1}.png".format(key, i * batch_size + idx),
                            ),
                        )
                        for idx in range(len(images))
                    )

                    output[key] = rgimages

            if i == 0:
                # Create an empty dict or list for the outputs
                outputs = defaultdict(list) if isinstance(output, Mapping) else []

            # Append the output
            if output is not None:
                if isinstance(output, Mapping):
                    for k in output.keys():
                        outputs[k].extend(output[k])
                else:
                    outputs.extend(output)

        # Reset the format
        if input_columns:
            self.set_format(previous_format)

        if not len(outputs):
            return None
        elif isinstance(outputs, dict):
            return dict(outputs)
        return outputs

    def filter(
        self,
        function: Optional[Callable] = None,
        with_indices=False,
        input_columns: Optional[Union[str, List[str]]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        drop_last_batch: bool = False,
        num_proc: Optional[int] = 64,
        **kwargs,
    ) -> Optional[VisionDataset]:
        """Apply a filter over the dataset."""
        # Just return if the function is None
        if function is None:
            logger.info("`function` None, returning None.")
            return None

        # Return if `self` has no examples
        if not len(self):
            logger.info("Dataset empty, returning None.")
            return None

        # Get some information about the function
        function_properties = self._inspect_function(
            function,
            with_indices,
            batched=batched,
        )
        assert function_properties.bool_output, "function must return boolean."

        # Map to get the boolean outputs and indices
        logger.info("Running `filter`, a new dataset will be returned.")
        outputs = self.map(
            function=function,
            with_indices=with_indices,
            input_columns=input_columns,
            batched=batched,
            batch_size=batch_size,
            drop_last_batch=drop_last_batch,
            num_proc=num_proc,
        )
        indices = np.where(outputs)[0]

        # Reset the format to set visible columns for the filter
        with self.format():
            # Filter returns a new dataset
            new_dataset = self.copy()
            new_dataset.set_visible_rows(indices)

        return new_dataset

    def to_dataloader(
        self,
        columns: Sequence[str],
        column_to_transform: Optional[Mapping[str, Callable]] = None,
        **kwargs,
    ) -> torch.utils.data.DataLoader:
        """Get a PyTorch dataloader that iterates over a subset of the columns
        (specified by `keys`) in the dataset. This is handy when using the dataset with
        training or evaluation loops outside of robustnessgym.  For example:
        ```
        dataset = Dataset(...)
        for img, target in dataset.to_dataloader(
            keys=["img_path", "label"],
            batch_size=16,
            num_workers=12
        ):
            out = model(img)
            loss = loss(out, target)
            ...
        ```

        Args:
            columns (Sequence[str]): A subset of the columns in the vision dataset.
                Specifies the columns to load. The dataloader will return values in same
                 order as `columns` here.
            column_to_transform (Optional[Mapping[str, Callable]], optional): A mapping
                from zero or more `keys` to callable transforms to be applied by the
                dataloader. Defaults to None, in which case no transforms are applied.
                Example: `column_to_transform={"img_path": transforms.Resize((16,16))}`.
                Note: these transforms will be applied after the transforms specified
                via the `transform` argument to `VisionDataset`.

        Returns:
            torch.utils.data.DataLoader: dataloader that iterates over dataset
        """
        img_folder = TorchDataset(
            self, columns=columns, column_to_transform=column_to_transform
        )
        return torch.utils.data.DataLoader(img_folder, **kwargs)

    def copy(self, deepcopy=False):
        """Return a copy of the dataset."""
        if deepcopy:
            return copy.deepcopy(self)
        else:
            dataset = VisionDataset()
            dataset.__dict__ = {k: copy.copy(v) for k, v in self.__dict__.items()}
            return dataset

    @classmethod
    def _state_keys(cls) -> set:
        """List of attributes that describe the state of the object."""
        return {
            "_data",
            "all_columns",
            "visible_rows",
            "_info",
            "_split",
            "_img_columns",
            "_updating_images",
            "_adding_images",
            "_callstack",
        }

    @classmethod
    def _assert_state_keys(cls, state: Dict) -> None:
        """Assert that a state contains all required keys."""
        assert (
            set(state.keys()) == cls._state_keys()
        ), f"State must contain all state keys: {cls._state_keys()}."

    def __getstate__(self) -> Dict:
        """Get the internal state of the dataset."""
        state = {key: getattr(self, key) for key in self._state_keys()}
        self._assert_state_keys(state)
        return state

    def __setstate__(self, state: Dict) -> None:
        """Set the internal state of the dataset."""
        if not isinstance(state, dict):
            raise ValueError(
                f"`state` must be a dictionary containing " f"{self._state_keys()}."
            )

        self._assert_state_keys(state)

        for key in self._state_keys():
            setattr(self, key, state[key])

        # Do some initialization
        self._initialize_state()

    @classmethod
    def load_from_disk(cls, path: str) -> VisionDataset:
        """Load the in-memory dataset from disk."""

        with gzip.open(os.path.join(path, "data.gz")) as f:
            dataset = pickle.load(f)
        # # Empty state dict
        # state = {}
        #
        # # Load the data
        # with gzip.open(os.path.join(path, "data.gz")) as f:
        #     state['_data'] = pickle.load(f)
        #
        # # Load the metadata
        # metadata = json.load(
        #     open(os.path.join(path, "metadata.json"))
        # )
        #
        # # Merge the metadata into the state
        # state = {**state, **metadata}

        # Create an empty `VisionDataset` and set its state
        # dataset = cls()
        # dataset.__setstate__(state)

        return dataset

    def save_to_disk(self, path: str):
        """Save the in-memory dataset to disk."""
        # Create all the directories to the path
        os.makedirs(path, exist_ok=True)

        # Store the data in a compressed format
        with gzip.open(os.path.join(path, "data.gz"), "wb") as f:
            pickle.dump(self, f)

        # # Get the dataset state
        # state = self.__getstate__()
        #
        # # Store the data in a compressed format
        # with gzip.open(os.path.join(path, "data.gz"), "wb") as f:
        #     pickle.dump(state['_data'], f)
        #
        # # Store the metadata
        # json.dump(
        #     {k: v for k, v in state.items() if k != '_data'},
        #     open(os.path.join(path, "metadata.json"), 'w'),
        # )


class TorchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: VisionDataset,
        columns: Sequence[str],
        column_to_transform: Optional[Mapping[str, Callable]] = None,
    ):
        """A torch dataset wrapper around `VisionDataset` that can be used in a
        torch dataloder.

        `VisionDataset.__getitem__` returns a dict containing `RGImage`
        objects, which is not compatible with the torch's default
        `collate_fn`. This dataset returns a subset of columns in the
        dataset specified by `keys` and calls `load` on any RGImage
        before returning to the `collate_fn`. It also supports
        specifying a different transformation function for each key
        returned.
        """
        self.columns = columns
        self.column_to_transform = (
            {} if column_to_transform is None else column_to_transform
        )
        self.dataset = dataset

    def __getitem__(self, index: int):
        row = self.dataset[index]
        vals = [
            self.column_to_transform.get(k, lambda x: x)(
                v.load() if isinstance(v, RGImage) else v
            )
            for k, v in ((key, row[key]) for key in self.columns)
        ]
        return tuple(vals) if len(vals) > 1 else vals[0]

    def __len__(self):
        return len(self.dataset)
