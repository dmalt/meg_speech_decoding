from functools import reduce
from typing import Callable

from ..signal import Signal, T

SignalProcessor = Callable[[Signal[T]], Signal[T]]


def compose_processors(*functions: SignalProcessor) -> SignalProcessor:
    return reduce(lambda f, g: lambda x: g(f(x)), functions)
