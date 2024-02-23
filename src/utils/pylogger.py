# pyright: reportIncompatibleMethodOverride=false
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pytorch_lightning.utilities.rank_zero import rank_prefixed_message, rank_zero_only

if TYPE_CHECKING:
    from collections.abc import Mapping


class RankedLogger(logging.LoggerAdapter):
    """A multi-GPU-friendly python command line logger."""

    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = False,
        extra: Mapping[str, object] | None = None,
    ) -> None:
        """Initializes a multi-GPU-friendly python command line logger that logs on all processes
        with their rank prefixed in the log message.

        :param name: The name of the logger. Default is ``__name__``.
        :param rank_zero_only: Whether to force all logs to only occur on the rank zero process. Default is `False`.
        :param extra: (Optional) A dict-like object which provides contextual information. See `logging.LoggerAdapter`.
        """
        logger = logging.getLogger(name)
        super().__init__(logger=logger, extra=extra)
        self.rank_zero_only = rank_zero_only

    def log(
        self,
        level: int,
        msg: str,
        rank: int | None = None,
        *args,
        **kwargs,
    ) -> None:
        """Delegate a log call to the underlying logger, after prefixing its message with the rank
        of the process it's being logged from. If `'rank'` is provided, then the log will only
        occur on that rank/process.

        :param level: The level to log at. Look at `logging.__init__.py` for more information.
        :param msg: The message to log.
        :param rank: The rank to log at.
        :param args: Additional args to pass to the underlying logging function.
        :param kwargs: Any additional keyword args to pass to the underlying logging function.
        """
        if self.isEnabledFor(level):
            current_rank = getattr(rank_zero_only, "rank", None)
            if current_rank is None:
                msg = "The `rank_zero_only.rank` needs to be set before use"
                raise RuntimeError(msg)

            msg, kwargs = self.process(msg, kwargs)
            msg = rank_prefixed_message(msg, current_rank)

            if (
                current_rank == 0
                if self.rank_zero_only
                else rank is None or current_rank == rank
            ):
                self.logger.log(level, msg, *args, **kwargs)
