import dill as pickle


class PicklerMixin:

    def __init__(self, *args, **kwargs):
        pass

    def save(self, path: str) -> None:
        """
        Save the object.
        """
        pickle.dump(self, open(path, 'wb'))

    @classmethod
    def load(cls, path: str):
        """
        Load the object from the path.
        """
        return pickle.load(open(path, 'rb'))