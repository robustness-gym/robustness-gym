from copy import deepcopy
from typing import Callable, Dict, List


def singlecolumn(func: Callable):
    """Assert that func is called with a single column.

    Mainly used with .apply(..) methods for CachedOperation and SliceBuilder.

    Args:
        func: function to wrap

    Returns: decorated function
    """

    def _singlecolumn(
        self, batch: Dict[str, List], columns: List[str], *args, **kwargs
    ):
        assert len(columns) == 1, "Must pass in a single column."
        return func(self, batch, columns, *args, **kwargs)

    return _singlecolumn


def prerequisites(*args):
    """
    Decorator to add a prerequisites attribute to any class.
    Args:
        *args: list of prerequisites

    Returns: a decorator
    """

    def _decorator(cls):
        _old_init = deepcopy(cls.__init__)

        def _new_init(self, *_args, **kwargs):
            _old_init(self, *_args, **kwargs)
            self.prerequisites = set(args)

        cls.__init__ = _new_init

        return cls

    return _decorator


def function_register():
    registry = {}

    def registrar(func):
        # Register the function
        registry[func.__name__] = func

        # Mark the fact that the function is decorated
        func.decorator = registrar

        return func

    registrar.all = registry
    return registrar


# Create processors that keep track of batch and dataset operations
batch_processing = function_register()
dataset_processing = function_register()


def methods_with_decorator(cls, decorator):
    """Returns all methods in cls with decorator as the outermost decorator.

    Credit: https://stackoverflow.com/questions/5910703/how-to-get-all-methods-of-a
    -python-class-with-given-decorator
    """
    for maybe_decorated in cls.__dict__.values():
        if hasattr(maybe_decorated, "decorator"):
            if maybe_decorated.decorator == decorator:
                yield maybe_decorated
