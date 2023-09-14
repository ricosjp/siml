
from typing import Optional
import abc


class ILoggingItem(metaclass=abc.ABCMeta):
    def __init__(
        self,
        val: int,
        title: str = None
    ) -> None:
        raise NotImplementedError()

    def __repr__(self) -> str:
        raise NotImplementedError()

    def format(
        self,
        *,
        formatter: str = None,
        padding_margin: int = None,
        key_orders: Optional[list[str]] = None,
        title: str = None,
        **kwards
    ) -> str:
        raise NotImplementedError()


class LoggingIntItem(ILoggingItem):
    def __init__(self, val: int, title: str, **kwards) -> None:
        assert isinstance(val, int)
        assert title is not None

        self._val = val
        self._title = title

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}: {self._val}, {self._title}'

    def format(
        self,
        *,
        formatter: str = None,
        padding_margin: int = None,
        title: str = None,
        **kwards
    ) -> str:
        if padding_margin is None:
            return str(self._val)

        if title is None:
            title = self._title

        str_size = len(title) + padding_margin
        return str(self._val).ljust(str_size, " ")


class LoggingFloatItem(ILoggingItem):
    def __init__(self, val: float, title: str) -> None:
        assert isinstance(val, float)
        assert title is not None

        self._val = val
        self._title = title

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}: {self._val}, {self._title}'

    def format(
        self,
        *,
        formatter: str = None,
        padding_margin: int = None,
        title: str = None,
        **kwards
    ) -> str:
        assert formatter is not None
        if padding_margin is None:
            return self._format_digit(formatter)
        else:
            return self._format_padding(
                formatter=formatter,
                padding_margin=padding_margin,
                title=title
            )

    def _format_digit(self, formatter: str) -> str:
        return f'{self._val:{formatter}}'

    def _format_padding(
        self, formatter: str, padding_margin: int, title: str = None
    ) -> str:

        if title is None:
            title = self._title
        val = f'{self._val:{formatter}}'
        str_size = len(self._title) + padding_margin
        return val.ljust(str_size, " ")


class LoggingDictItem(ILoggingItem):
    def __init__(
        self,
        val: dict[str, float],
        title: str
    ) -> None:
        assert isinstance(val, dict)
        assert title is not None

        self._val = val
        self._title = title

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}: {self._val}, {self._title}'

    def format(
        self,
        *,
        formatter: str = None,
        padding_margin: int = None,
        key_orders: Optional[str] = None,
        title: str = None,
        **kwards
    ) -> str:
        assert formatter is not None

        if len(self._val) == 0:
            return ""

        keys = self._val.keys() if key_orders is None else key_orders
        for key in keys:
            if key not in self._val:
                raise KeyError(f"{key} is not found in self._val")

        if padding_margin is None:
            return self._format_digits(formatter=formatter, keys=keys)
        else:
            return self._format_padding(
                formatter=formatter,
                padding_margin=padding_margin,
                keys=keys,
                title=title
            )

    def _format_padding(
        self,
        formatter: str,
        padding_margin: int,
        keys: list[str],
        title: str = None
    ) -> str:
        vals = [
            self._format_each_padding(
                k, self._val[k], formatter, padding_margin,
                title=title
            )
            for k in keys
        ]
        return "".join(vals)

    def _format_each_padding(
        self,
        key: str,
        value: float,
        formatter: str,
        padding_margin: int,
        title: str = None
    ) -> str:
        if title is None:
            title = self._title
        str_size = len(title + key) + padding_margin
        v = f'{value:{formatter}}'
        return v.ljust(str_size, " ")

    def _format_digits(self, formatter: str, keys: list[str]) -> str:
        vals = [f"{self._val[k]:{formatter}}" for k in keys]
        return ", ".join(vals)


class LoggingStrItem(ILoggingItem):
    def __init__(self, val: str, **kwards) -> None:
        assert isinstance(val, str)
        self._val = val
        self._title = val

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}: {self._val}, {self._title}'

    def format(
        self,
        *,
        formatter: str = None,
        padding_margin: int = None,
        title: str = None,
        **kwards
    ) -> str:
        if padding_margin is None:
            return self._val
        else:
            return self._format_padding(padding_margin, title=title)

    def _format_padding(
        self, padding_margin: int, title: str = None
    ) -> str:
        if title is None:
            title = self._title
        str_size = len(title) + padding_margin
        return self._val.ljust(str_size, " ")
