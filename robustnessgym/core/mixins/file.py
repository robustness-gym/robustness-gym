from pathlib import Path
from typing import Union


class FileMixin:
    """Mixin for adding in single filepath."""

    def __init__(self, filepath: Union[str, Path], *args, **kwargs):
        super(FileMixin, self).__init__(*args, **kwargs)

        if isinstance(filepath, str):
            filepath = Path(filepath)

        # Assign the path
        self.filepath = filepath

    @property
    def filename(self):
        # TODO:
        return self.filepath.name

    @property
    def extension(self):
        # TODO:
        return self.filepath.extension
