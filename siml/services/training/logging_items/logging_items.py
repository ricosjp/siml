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
        padding_margin: int = None
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
        padding_margin: int = None
    ) -> str:
        if padding_margin is None:
            return str(self._val)

        str_size = len(self._title) + padding_margin
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
        padding_margin: int = None
    ) -> str:
        assert formatter is not None
        if padding_margin is None:
            return self._format_digit(formatter)
        else:
            return self._format_padding(
                formatter=formatter,
                padding_margin=padding_margin
            )

    def _format_digit(self, formatter: str) -> str:
        return f'{self._val:{formatter}}'

    def _format_padding(self, formatter: str, padding_margin: int) -> str:
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
        padding_margin: int = None
    ) -> str:
        assert formatter is not None
        if padding_margin is None:
            return self._format_digits(formatter=formatter)
        else:
            return self._format_padding(
                formatter=formatter,
                padding_margin=padding_margin
            )

    def _format_padding(self, formatter: str, padding_margin: int) -> str:
        vals = [
            self._format_each_padding(k, v, formatter, padding_margin)
            for k, v in self._val.items()
        ]
        return "".join(vals)

    def _format_each_padding(
        self,
        key: str,
        value: float,
        formatter: str,
        padding_margin: int
    ) -> str:
        str_size = len(self._title + key) + padding_margin
        v = f'{value:{formatter}}'
        return v.ljust(str_size, " ")

    def _format_digits(self, formatter: str) -> str:
        vals = [f"{v:{formatter}}" for v in self._val.values()]
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
        padding_margin: int = None
    ) -> str:
        if padding_margin is None:
            return self._val
        else:
            return self._format_padding(padding_margin)

    def _format_padding(self, padding_margin: int) -> str:
        str_size = len(self._title) + padding_margin
        return self._val.ljust(str_size, " ")
