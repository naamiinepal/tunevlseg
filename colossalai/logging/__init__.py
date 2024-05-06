import logging

from .logger import DistributedLogger

__all__ = ["DistributedLogger", "disable_existing_loggers", "get_dist_logger"]


def get_dist_logger(name: str = "colossalai") -> DistributedLogger:
    """Get logger instance based on name. The DistributedLogger will create singleton instances,
    which means that only one logger instance is created per name.

    Args:
        name (str): name of the logger, name must be unique

    Returns:
        :class:`colossalai.logging.DistributedLogger`: A distributed logger singleton instance.
    """
    return DistributedLogger.get_instance(name=name)


def disable_existing_loggers(
    include: list[str] | None = None, exclude: list[str] | None = None
) -> None:
    """Set the level of existing loggers to `WARNING`. By default, it will "disable" all existing loggers except the logger named "colossalai".

    Args:
        include (Optional[List[str]], optional): Loggers whose name in this list will be disabled.
            If set to `None`, `exclude` argument will be used. Defaults to None.
        exclude (List[str], optional): Loggers whose name not in this list will be disabled.
            This argument will be used only when `include` is None. Defaults to ['colossalai'].
    """
    if exclude is None:
        exclude = ["colossalai"]
    if include is None:

        def filter_func(name):
            return name not in exclude
    else:

        def filter_func(name):
            return name in include

    for log_name in logging.Logger.manager.loggerDict:
        if filter_func(log_name):
            logging.getLogger(log_name).setLevel(logging.WARNING)
