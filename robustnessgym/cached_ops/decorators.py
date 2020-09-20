from typing import Callable, Dict, List


def singlecolumn(func: Callable):
    """
    Assert that func is called with a single column.

    Mainly used with .apply(..) methods for CachedOperation and SliceBuilder.

    Args:
        func: function to wrap

    Returns: decorated function

    """

    def _singlecolumn(self,
                      batch: Dict[str, List],
                      columns: List[str],
                      *args,
                      **kwargs):
        assert len(columns) == 1, "Must pass in a single column."
        return func(self, batch, columns, *args, **kwargs)

    return _singlecolumn
