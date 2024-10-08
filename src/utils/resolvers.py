from __future__ import annotations

import functools
from importlib import import_module
from typing import TYPE_CHECKING, ParamSpec, TypeVar

from omegaconf import OmegaConf

if TYPE_CHECKING:
    from collections.abc import Callable

    T = TypeVar("T")
    P = ParamSpec("P")


def import_resolver(string: str):
    """Import and resolve the string.

    Args:
    ----
        string (str): The string to import and resolve.

    Raises:
    ------
        ValueError: If the string is not a module path.

    Returns:
    -------
        Any: The imported and resolved object.

    """
    splitted = string.split(".", 1)
    if len(splitted) != 2:
        msg = "The string must be a module path"
        raise ValueError(msg)

    module, n = splitted

    # Use relative import if the module is empty
    if not module:
        module = string

    imported_module = import_module(module, package="src")

    for attr in n.split("."):
        imported_module = getattr(imported_module, attr)

    return imported_module


def register_new_resolvers(func: Callable[P, T]) -> Callable[P, T]:
    """Register new resolvers for omegaconf.

    Args:
    ----
        func: The function that instantiates the objects

    Returns:
    -------
        The same fuction passed but with new omegaconf resolvers registered

    """

    @functools.wraps(func)
    def inner_func(*args: P.args, **kwargs: P.kwargs) -> T:
        # Register a resolver to evaluate in the yaml file
        OmegaConf.register_new_resolver("literal_eval", eval, replace=True)

        # Register a resolver to import a function in the yaml file and evaluate it
        # It can be done only for those dtypes which are serializable
        OmegaConf.register_new_resolver(
            "import_eval", import_resolver, replace=True, use_cache=True
        )

        return func(*args, **kwargs)

    return inner_func
