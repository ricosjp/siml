
from ignite.metrics.metric import Metric, reinit__is_reduced

from siml.loss_operations import ILossCalculator


class LossDetailsMetrics(Metric):
    def __init__(self, loss_calculator: ILossCalculator) -> None:
        self._loss_calculator = loss_calculator
        self._results = None
        super(LossDetailsMetrics, self).__init__()

    @reinit__is_reduced
    def reset(self):
        self._results = {}
        return

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output[0].detach(), output[1].detach()

        loss_details = self._loss_calculator.calculate_loss_details(
            y_pred, y
        )

        for k, v in loss_details.items():
            if k in self._results:
                self._results[k] = v.item()
            else:
                self._results[k] += v.item()
        return

    def compute(self) -> dict[str, float]:
        return self._results
