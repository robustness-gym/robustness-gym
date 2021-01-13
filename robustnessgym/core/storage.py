import dill


class StorageMixin:
    """Mixin class for serialization."""

    def __init__(self, *args, **kwargs):
        super(StorageMixin, self).__init__(*args, **kwargs)

    def save(self, path: str) -> None:
        """Save the object."""
        dill.dump(self, open(path, "wb"))

    @classmethod
    def load(cls, path: str) -> object:
        """Load the object from the path."""
        return dill.load(open(path, "rb"))
