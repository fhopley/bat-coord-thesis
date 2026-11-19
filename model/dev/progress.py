"""A console progress line for long single-threaded runs.

A rolling optimisation over a site-year is minutes of silent work, and a sweep
is dozens of those. Without feedback there is no way to tell a slow run from a
hung one, which matters most exactly when something has gone wrong.

Deliberately dependency-free and deliberately dumb: it writes a carriage-return
line to stderr and gets out of the way. stderr rather than stdout so that
redirecting a report to a file does not capture progress spam, and so progress
survives when stdout is piped.
"""
from __future__ import annotations

import sys
import time


class ConsoleProgress:
    """Callable matching rolling_optimisation's progress hook.

    Usage:
        with ConsoleProgress("S0008 horizon 24h") as p:
            rolling_optimisation(..., progress=p)

    Rate-limited by wall clock rather than by call count, so the update rate is
    the same whether an interval takes 1 ms or 10.
    """

    def __init__(self, label: str, min_seconds: float = 0.5,
                 stream=None, enabled: bool = True):
        self.label = label
        self.min_seconds = min_seconds
        self.stream = stream if stream is not None else sys.stderr
        self.enabled = enabled and self.stream.isatty()
        self._t0 = time.perf_counter()
        self._last = 0.0
        self._width = 0

    def __enter__(self) -> "ConsoleProgress":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __call__(self, done: int, total: int) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        final = done >= total
        if not final and now - self._last < self.min_seconds:
            return
        self._last = now
        elapsed = now - self._t0
        frac = done / total if total else 1.0
        eta = (elapsed / frac - elapsed) if frac > 1e-6 else float('nan')
        line = (f"  {self.label}: {frac:5.1%}  {done:,}/{total:,}  "
                f"elapsed {_hms(elapsed)}  eta {_hms(eta)}")
        self._width = max(self._width, len(line))
        self.stream.write("\r" + line.ljust(self._width))
        self.stream.flush()

    def close(self) -> None:
        """Clear the line so the next print starts clean."""
        if self.enabled and self._width:
            self.stream.write("\r" + " " * self._width + "\r")
            self.stream.flush()
            self._width = 0


def _hms(seconds: float) -> str:
    if seconds != seconds or seconds == float('inf'):    # NaN or inf
        return "--:--"
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
