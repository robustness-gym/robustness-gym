"""File containing abstract base class for datasets."""
import abc

from datasets.arrow_dataset import DatasetInfoMixin


class AbstractDataset(
    abc.ABC,
    DatasetInfoMixin,
):
    """An abstract dataset class."""

    def __init__(self, *args, **kwargs):
        super(AbstractDataset, self).__init__(*args, **kwargs)

    def __repr__(self):
        return (
            f"RobustnessGym{self.__class__.__name__}"
            f"(num_rows: {self._dataset.num_rows})"
        )

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def num_rows(self):
        """Number of total rows in the dataset."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def columns(self):
        """Columns in the dataset."""
        raise NotImplementedError
