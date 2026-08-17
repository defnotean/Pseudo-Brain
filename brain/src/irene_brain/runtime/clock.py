"""Monotonic clock primitives for deterministic and real-time execution.

The production clock exposes native Windows ``QueryPerformanceCounter`` ticks
and their frequency.  Other platforms, and Windows hosts where QPC cannot be
initialized, use ``time.perf_counter_ns`` with a one-gigahertz tick frequency.
Neither backend reads wall-clock time.
"""

from __future__ import annotations

import os
import time
import ctypes
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

NANOSECONDS_PER_SECOND = 1_000_000_000


def _require_plain_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def ticks_to_nanoseconds(ticks: int, frequency_hz: int) -> int:
    """Convert a raw tick count or delta to nanoseconds without floats."""

    ticks = _require_plain_int(ticks, name="ticks")
    frequency_hz = _require_plain_int(frequency_hz, name="frequency_hz")
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive")
    return ticks * NANOSECONDS_PER_SECOND // frequency_hz


def nanoseconds_to_ticks(nanoseconds: int, frequency_hz: int) -> int:
    """Convert nanoseconds to raw ticks, rounding toward negative infinity."""

    nanoseconds = _require_plain_int(nanoseconds, name="nanoseconds")
    frequency_hz = _require_plain_int(frequency_hz, name="frequency_hz")
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive")
    return nanoseconds * frequency_hz // NANOSECONDS_PER_SECOND


@runtime_checkable
class ClockProtocol(Protocol):
    """The minimal injectable clock required by the continuous driver."""

    @property
    def frequency_hz(self) -> int:
        """Return the number of raw clock ticks per second."""
        ...

    def now_ticks(self) -> int:
        """Return a monotonic raw tick reading."""
        ...


class QueryPerformanceClock:
    """System monotonic clock with native QPC readings when available.

    QPC is selected once during construction.  The backend is never changed
    after readings begin because switching tick domains would make recorded
    durations invalid.  Pass ``force_fallback=True`` for a deterministic way
    to exercise the safe ``perf_counter_ns`` backend in tests.
    """

    __slots__ = ("_backend_name", "_frequency_hz", "_read_ticks")

    def __init__(self, *, force_fallback: bool = False) -> None:
        if not isinstance(force_fallback, bool):
            raise TypeError("force_fallback must be a bool")

        self._backend_name = "perf_counter_ns"
        self._frequency_hz = NANOSECONDS_PER_SECOND
        self._read_ticks: Callable[[], int] = time.perf_counter_ns

        if force_fallback or os.name != "nt":
            return

        try:
            win_dll: Any = getattr(ctypes, "WinDLL")
            kernel32: Any = win_dll("kernel32", use_last_error=True)
            query_frequency: Any = kernel32.QueryPerformanceFrequency
            query_frequency.argtypes = [ctypes.POINTER(ctypes.c_longlong)]
            query_frequency.restype = ctypes.c_int
            query_counter: Any = kernel32.QueryPerformanceCounter
            query_counter.argtypes = [ctypes.POINTER(ctypes.c_longlong)]
            query_counter.restype = ctypes.c_int

            frequency = ctypes.c_longlong()
            if not query_frequency(ctypes.byref(frequency)) or frequency.value <= 0:
                return

            # Confirm the counter is readable before committing to its domain.
            initial = ctypes.c_longlong()
            if not query_counter(ctypes.byref(initial)) or initial.value < 0:
                return
        except (AttributeError, OSError):
            return

        def read_qpc() -> int:
            value = ctypes.c_longlong()
            if not query_counter(ctypes.byref(value)):
                # Do not silently change clock domains after initialization.
                raise RuntimeError("QueryPerformanceCounter failed after initialization")
            return int(value.value)

        self._backend_name = "query_performance_counter"
        self._frequency_hz = int(frequency.value)
        self._read_ticks = read_qpc

    @property
    def frequency_hz(self) -> int:
        return self._frequency_hz

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def uses_query_performance_counter(self) -> bool:
        return self._backend_name == "query_performance_counter"

    def now_ticks(self) -> int:
        return self._read_ticks()

    def now_ns(self) -> int:
        return ticks_to_nanoseconds(self.now_ticks(), self.frequency_hz)


class ManualClock:
    """A deterministic, manually advanced clock for unit and replay tests."""

    __slots__ = ("_frequency_hz", "_fractional_numerator", "_ticks")

    def __init__(self, *, frequency_hz: int = NANOSECONDS_PER_SECOND, ticks: int = 0) -> None:
        frequency_hz = _require_plain_int(frequency_hz, name="frequency_hz")
        ticks = _require_plain_int(ticks, name="ticks")
        if frequency_hz <= 0:
            raise ValueError("frequency_hz must be positive")
        if ticks < 0:
            raise ValueError("ticks cannot be negative")
        self._frequency_hz = frequency_hz
        self._ticks = ticks
        self._fractional_numerator = 0

    @property
    def frequency_hz(self) -> int:
        return self._frequency_hz

    @property
    def backend_name(self) -> str:
        return "manual"

    def now_ticks(self) -> int:
        return self._ticks

    def now_ns(self) -> int:
        return ticks_to_nanoseconds(self._ticks, self._frequency_hz)

    def set_ticks(self, ticks: int) -> int:
        ticks = _require_plain_int(ticks, name="ticks")
        if ticks < self._ticks:
            raise ValueError("a monotonic clock cannot move backwards")
        self._ticks = ticks
        self._fractional_numerator = 0
        return self._ticks

    def advance_ticks(self, ticks: int = 1) -> int:
        ticks = _require_plain_int(ticks, name="ticks")
        if ticks < 0:
            raise ValueError("ticks cannot be negative")
        self._ticks += ticks
        return self._ticks

    def advance_ns(self, nanoseconds: int) -> int:
        """Advance by a duration while retaining sub-tick remainder exactly."""

        nanoseconds = _require_plain_int(nanoseconds, name="nanoseconds")
        if nanoseconds < 0:
            raise ValueError("nanoseconds cannot be negative")
        numerator = (
            nanoseconds * self._frequency_hz + self._fractional_numerator
        )
        delta_ticks, self._fractional_numerator = divmod(
            numerator,
            NANOSECONDS_PER_SECOND,
        )
        self._ticks += delta_ticks
        return self._ticks


# Concise aliases for callers that prefer the platform or testing terminology.
QpcClock = QueryPerformanceClock
FakeClock = ManualClock
SystemClock = QueryPerformanceClock


__all__ = [
    "ClockProtocol",
    "FakeClock",
    "ManualClock",
    "NANOSECONDS_PER_SECOND",
    "QpcClock",
    "QueryPerformanceClock",
    "SystemClock",
    "nanoseconds_to_ticks",
    "ticks_to_nanoseconds",
]
