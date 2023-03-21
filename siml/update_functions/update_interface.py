import abc
from typing import Any
import torch

from siml.networks.network import Network


class IStepUpdateFunction(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            model: Network,
            optimizer: torch.optim.Optimizer,
            *args: Any,
            **kwds: Any) -> float:
        raise NotImplementedError()
