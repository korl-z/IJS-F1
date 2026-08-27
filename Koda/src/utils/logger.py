"""Small logging helpers that coexist cleanly with tqdm progress bars."""

from __future__ import annotations

import copy
import logging
import os
import sys
from typing import Any, Iterable, Optional, TypeVar

from tqdm.auto import tqdm

T = TypeVar("T")

CLR = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[97;41m",
    "RESET": "\033[0m",
    "BOLD": "\033[1m",
    "TIME": "\033[90m",
}


def _has_color(file: Any) -> bool:
    """Return whether ANSI colors are appropriate for an output stream."""
    if os.getenv("NO_COLOR") is not None:
        return False
    if os.getenv("FORCE_COLOR") is not None:
        return True
    if os.getenv("TERM", "").lower() == "dumb":
        return False
    return bool(getattr(file, "isatty", lambda: False)())


class TqdmHandler(logging.Handler):
    """Write log records without overwriting an active tqdm bar."""

    _is_tqdm = True

    def __init__(self, file: Any = None) -> None:
        super().__init__()
        self.file = file if file is not None else sys.stderr

    def emit(self, rec: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(rec), file=self.file)
        except Exception:
            self.handleError(rec)

    def flush(self) -> None:
        if hasattr(self.file, "flush"):
            self.file.flush()


class LogFormatter(logging.Formatter):
    """Add a compact timestamp and a colored level tag."""

    def __init__(self, color: bool = True) -> None:
        rst = CLR["RESET"] if color else ""
        gry = CLR["TIME"] if color else ""
        fmt = f"{gry}[%(asctime)s]{rst} %(tag)s %(message)s"
        super().__init__(fmt=fmt, datefmt="%H:%M:%S")
        self.color = color

    def format(self, rec: logging.LogRecord) -> str:
        row = copy.copy(rec)
        lvl = f"[{rec.levelname:<8}]"
        if self.color:
            clr = CLR.get(rec.levelname, CLR["RESET"])
            row.tag = f"{clr}{CLR['BOLD']}{lvl}{CLR['RESET']}"
        else:
            row.tag = lvl
        return super().format(row)


def setup_logger(
    name: str = "app",
    level: int = logging.INFO,
    *,
    color: Optional[bool] = None,
    file: Any = None,
) -> logging.Logger:
    """Create or update one tqdm-safe logger."""
    out = file if file is not None else sys.stderr
    lg = logging.getLogger(name)
    lg.setLevel(level)
    lg.propagate = False

    hd = next((x for x in lg.handlers if getattr(x, "_is_tqdm", False)), None)
    if hd is None:
        hd = TqdmHandler(out)
        lg.addHandler(hd)
    else:
        hd.file = out

    hd.setLevel(level)
    hd.setFormatter(LogFormatter(_has_color(out) if color is None else color))
    return lg


def tqdm_bar(
    it: Optional[Iterable[T]] = None,
    desc: str = "Processing",
    total: Optional[int] = None,
    **kw: Any,
) -> tqdm:
    """Return a compact progress bar with safe defaults."""
    opt = {
        "desc": desc,
        "total": total,
        "dynamic_ncols": True,
        "leave": True,
        "mininterval": 0.1,
        "smoothing": 0.1,
        "bar_format": (
            "{l_bar}{bar:24} {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        ),
    }
    opt.update(kw)
    return tqdm(it, **opt)


logger = setup_logger()
