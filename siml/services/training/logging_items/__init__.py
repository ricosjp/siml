from typing import Union

from .logging_items import (ILoggingItem, LoggingDictItem, LoggingFloatItem,
                            LoggingIntItem, LoggingStrItem)


def create_logitems(
    value: Union[str, float, int, dict[str, float]],
    title: str = None
) -> ILoggingItem:

    if isinstance(value, str):
        return LoggingStrItem(val=value)

    if isinstance(value, float):
        return LoggingFloatItem(
            val=value, title=title)

    if isinstance(value, int):
        return LoggingIntItem(
            val=value, title=title
        )

    if isinstance(value, dict):
        return LoggingDictItem(
            val=value, title=title
        )

    raise NotImplementedError(
        f"{type(value)} is not implemented as a logging item."
    )
