import pydantic.dataclasses as dc


@dc.dataclass(init=False, frozen=True)
class SimlConstItems:
    DEPLOYED_MODEL_NAME: str = "DEPLOYED"
