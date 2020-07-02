import dill as pickle


class StorageMixin:
    def __init__(self, *args, **kwargs):
        super(StorageMixin, self).__init__(*args, **kwargs)

    def save(self, path: str) -> None:
        """Save the object."""
        pickle.dump(self, open(path, "wb"))

    @classmethod
    def load(cls, path: str):
        """Load the object from the path."""
        return pickle.load(open(path, "rb"))
