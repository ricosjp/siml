
from ignite.metrics.metric import Metric, reinit__is_reduced

from siml.loss_operations import ILossCalculator


class LossDetailsMetrics(Metric):
    def __init__(self, loss_calculator: ILossCalculator) -> None:
        self._loss_calculator = loss_calculator
        self._results = None
        self._num_examples = None
        super(LossDetailsMetrics, self).__init__()

    @reinit__is_reduced
    def reset(self):
        self._results = {}
        self._num_examples = 0
        return

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output[0], output[1]

        loss_details = self._loss_calculator.calculate_loss_details(
            y_pred, y
        )
        self._num_examples += 1
        for k, v in loss_details.items():
            if k not in self._results:
                self._results[k] = v.item()
            else:
                self._results[k] += v.item()
        return

    def compute(self) -> dict[str, float]:
        if self._num_examples == 0:
            raise ValueError("Cannot calculate loss")
        _results = {
            k: v / self._num_examples for k, v in self._results.items()
        }
        return _results
