from __future__ import annotations

import logging
from functools import wraps
from time import perf_counter
from typing import Any, Callable, Generator, Generic, Iterable, TypeVar

T = TypeVar("T", covariant=True)


def infinite(x: Iterable[T]) -> Generator[T, None, None]:
    while True:
        iterator = iter(x)
        for item in iterator:
            yield item


class limited(Generic[T]):
    def __init__(self, x: Iterable[T]):
        self.x = x

    def by(self, y: Iterable) -> Generator[T, None, None]:
        for x_out, _ in zip(self.x, y):
            yield x_out


RT = TypeVar("RT")


class log_execution_time:
    """
    Log start (optionally) and execution time of the decorated function

    Parameters
    ----------
    start_message: str | None, default=None
        If not None, log this message at the start of function execution

    Notes
    -----
    Messages are logged with level=logging.INFO
    For each function the new logger with name = <module_name>.<funciton_name>
    is created

    """
    def __init__(self, desc: str = "") -> None:
        self.desc = " " + desc if desc else desc

    def __call__(self, func: Callable[..., RT]) -> Callable[..., RT]:
        logger = logging.getLogger(func.__module__ + "." + func.__name__)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: dict[str, Any]) -> RT:
            logger.info("Started" + self.desc)
            t1 = perf_counter()
            res = func(*args, **kwargs)
            logger.info(f"Finished{self.desc} in {perf_counter() - t1:.2f} sec")
            return res

        return wrapper
