from typing import Collection, Dict
from dataclasses import dataclass

from robustnessgym.core.tools import nested_map


@dataclass
class StateClass:
    klass: type
    state: object


class StateDictMixin:
    def __init__(self, *args, **kwargs):
        super(StateDictMixin, self).__init__(*args, **kwargs)

    @classmethod
    def _state_keys(cls) -> Collection:
        raise NotImplementedError

    def get_state(self) -> Dict:
        """Get the internal state of the dataset."""
        def _apply_get_state(obj):
            if hasattr(obj, "get_state"):
                return StateClass(**{"klass": type(obj), "state": obj.get_state()})
            else:
                return obj

        state = nested_map(
            _apply_get_state, {key: getattr(self, key) for key in self._state_keys()}
        )
        self._assert_state_keys(state)
        return state

    @classmethod
    def _assert_state_keys(cls, state: Dict) -> None:
        """Assert that a state contains all required keys."""
        assert (
            set(state.keys()) == cls._state_keys()
        ), f"State must contain all state keys: {cls._state_keys()}."

    @classmethod
    def from_state(cls, state: Dict, *args, **kwargs):
        """Set the internal state of the dataset."""
        cls._assert_state_keys(state)

        def _apply_from_state(obj):
            if isinstance(obj, StateClass):
                return obj.klass.from_state(obj.state, *args, **kwargs)
            else:
                return obj

        state = nested_map(_apply_from_state, state)
        obj = cls()
        obj.__dict__.update(state)
        return obj
