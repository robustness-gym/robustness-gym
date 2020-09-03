import hashlib
import json
from functools import partial
from typing import Mapping, Sequence, List

import cytoolz as tz


def recmerge(*objs, merge_sequences=False):
    """
    Recursively merge an arbitrary number of collections.
    For conflicting values, later collections to the right are given priority.
    By default (merge_sequences=False), sequences are treated as a normal value and not merged.
    """
    if isinstance(objs, tuple) and len(objs) == 1:
        # A squeeze operation since merge_with generates tuple(list_of_objs,)
        objs = objs[0]
    if all([isinstance(obj, Mapping) for obj in objs]):
        # Merges all the collections, recursively applies merging to the combined values
        return tz.merge_with(partial(recmerge, merge_sequences=merge_sequences), *objs)
    elif all([isinstance(obj, Sequence) for obj in objs]) and merge_sequences:
        # Merges sequence values by concatenation
        return list(tz.concat(objs))
    else:
        # If colls does not contain mappings, simply pick the last one
        return tz.last(objs)


def persistent_hash(s: str):
    """
    Compute a hash that persists across multiple Python sessions for a string.
    """
    return int(hashlib.sha224(s.encode()).hexdigest(), 16)


def strings_as_json(strings: List[str]):
    """
    Convert a list of strings to JSON.

    Args:
        strings: A list of str.

    Returns: JSON dump of the strings.

    """
    return json.dumps(strings) if len(strings) > 1 else strings[0]
