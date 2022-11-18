from typing import (
    List,
    Tuple,
    Union,
    Dict,
    Any,
    Optional,
    Generic,
    TypeVar,
    Type,
    Generator,
)
import contextlib
import torch
import numpy
from allennlp.common.registrable import Registrable
import logging

logger = logging.getLogger(__name__)

valueT = TypeVar(
    "valueT", bound=Union[float, torch.Tensor, numpy.ndarray, List[float]]
)

returnT = TypeVar(
    "returnT",
    bound=Union[float, torch.Tensor, numpy.ndarray, List[float], None],
)


def _combine_keys(k1: str, k2: str) -> str:
    if k1 and k2:
        return ".".join([k1, k2])
    elif k1:
        return k1
    elif k2:
        return k2
    else:
        return ""


class LoggedValue(Generic[valueT, returnT]):
    def __init__(self) -> None:
        self.skip_logging = False

    def log(
        self,
        value: valueT,
    ) -> None:
        raise NotImplementedError

    def get(self, reset: bool = False) -> Optional[returnT]:
        raise NotImplementedError


class LoggedScalarScalar(LoggedValue[float, float]):
    """
    Keeps running average of a scalar.
    """

    def __init__(self, init: float = 0):
        super().__init__()
        self.value = 0.0
        self.steps = 0

    def log(self, value: float) -> None:
        if not self.skip_logging:
            self.accumulate(value)

    def accumulate(self, value: float) -> None:
        self.value += value
        self.steps += 1

    def reset(self) -> None:
        self.value = 0.0
        self.steps = 0

    def reduce(self) -> Optional[float]:
        if self.steps <= 0:
            return None
        assert self.steps > 0

        return self.value / self.steps

    def get(self, reset: bool = False) -> Optional[float]:
        result = self.reduce()

        if reset:
            self.reset()

        return result


class LoggedScalarScalarSample(LoggedScalarScalar):
    """
    Keeps the latest sample of a scalar.
    """

    def get(self, reset: bool = True) -> Optional[float]:
        result = self.reduce()
        reset = True

        if reset:
            self.reset()

        return result


class LoggedNPArrayNPArraySample(
    LoggedValue[Union[torch.Tensor, numpy.ndarray], Optional[numpy.ndarray]]
):
    """
    Keeps the latest sample of numpy array.
    """

    def __init__(self, init: float = 0):
        super().__init__()
        self.value: Optional[numpy.ndarray] = None

    def accumulate(self, value: Union[torch.Tensor, numpy.ndarray]) -> None:
        if isinstance(value, torch.Tensor):
            value_ = value.detach().cpu().numpy()
        self.value = value_

    def reset(self) -> None:
        self.value = None

    def reduce(self) -> Optional[numpy.ndarray]:
        return self.value

    def get(self, reset: bool = True) -> Optional[numpy.ndarray]:
        result = self.reduce()
        reset = True

        if reset:
            self.reset()

        return result


LoggedValueT = TypeVar("LoggedValueT", bound=LoggedValue)


class LoggingMixin(object):
    """
    Adds an attribute (dict) that holds various scalar and vector values.
    It also adds methods to add and remove information from this attribute.

    Use `self.logging_children.append()` to add child modules that are instances of `LoggingMixin`
    and that to want to show in logs.
    """

    def __init__(self, log_key: str = "", **kwargs: Any) -> None:
        # we need to accept and pass **kwargs to enable mulitple inheritance
        # required to use this class as mixin
        super().__init__(**kwargs)  # type: ignore
        self.logging_buffer: Dict[str, LoggedValue] = {}
        self.logging_children: List[LoggingMixin] = []
        self.log_key = log_key
        self.skip_logging = False
        self._skip_logging_cache: Optional[bool] = None

    def skip_logging_recursively(self) -> None:
        self.skip_logging = True

        for child in self.logging_children:
            child.skip_logging_recursively()

    def cache_logging_state_recursively(self) -> None:
        self.cache_logging_state()

        for child in self.logging_children:
            child.cache_logging_state_recursively()

    def cache_logging_state(self) -> None:
        assert (
            self._skip_logging_cache is None
        ), f"Cannot set logging cache for {self} without reseting the previous cache"
        self._skip_logging_cache = self.skip_logging

    def restore_logging_state(self) -> None:
        assert (
            self._skip_logging_cache is not None
        ), f"The cache was never set for {self}"
        self.skip_logging = self._skip_logging_cache
        self._skip_logging_cache = None

    def restore_logging_state_recursively(self) -> None:
        self.restore_logging_state()

        for child in self.logging_children:
            child.restore_logging_state_recursively()

    def remove_key_from_logging_buffer(self, key: str) -> None:
        self.logging_buffer.pop(key)

    def clear_logging_buffer(self) -> None:
        self.logging_buffer = {}

    def clear_logging_children(self) -> None:
        self.logging_children = []

    def log(self, key: str, value: valueT) -> None:
        if not self.skip_logging:
            self.logging_buffer[key].log(value)

    def get(self, key: str, reset: bool = False) -> Optional[valueT]:
        return self.logging_buffer[key].get(reset=reset)

    def get_all(
        self,
        reset: bool = True,
        type_: Optional[Tuple[Type[LoggedValueT], ...]] = None,
    ) -> Dict[str, valueT]:
        self_values = {}

        def take(v: LoggedValue) -> bool:
            return isinstance(v, type_) if type_ is not None else True

        try:
            self_values_ = {
                _combine_keys(self.log_key, k): v.get(reset=reset)
                for k, v in self.logging_buffer.items()
                if take(v)
            }
            # drop None values
            self_values = {
                k: v for k, v in self_values_.items() if v is not None
            }

        except Exception as e:
            breakpoint()
        children_values = {  # type: ignore[var-annotated]
            _combine_keys(self.log_key, k): v
            for child in self.logging_children
            for k, v in child.get_all(reset=reset, type_=type_).items()
        }

        return {**self_values, **children_values}

    @contextlib.contextmanager
    def no_logging(self) -> Generator[None, None, None]:
        self.cache_logging_state_recursively()

        try:
            self.skip_logging_recursively()
            yield
        finally:
            self.restore_logging_state_recursively()
