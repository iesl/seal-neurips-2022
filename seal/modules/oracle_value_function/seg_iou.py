from typing import Any, Optional
import torch
from seal.modules.oracle_value_function import OracleValueFunction


@OracleValueFunction.register("seg-iou")
class SegIoUValueFunction(OracleValueFunction):
    """Return oracle value that is based on segmentation IoU. """

    @property
    def upper_bound(self) -> float:
        return 1.0

    def compute(
        self,
        labels: torch.Tensor, # (b*n, 1, 24, 24) where n is from expansion
        y_hat: torch.Tensor, # (b*n, 1, 24, 24)
        mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:

        y_hat = y_hat.reshape(-1, y_hat.size()[-2] * y_hat.size()[-1])  # shape (batch*num_sample, pixel)
        labels = labels.reshape(-1, labels.size()[-2] * labels.size()[-1])
        intersect = torch.sum(torch.min(y_hat, labels), dim=-1)
        union = torch.sum(torch.max(y_hat, labels), dim=-1)

        # epsilon = torch.full(union.size(), 10**-8).to(union.device)
        # iou = intersect / torch.max(epsilon, union)
        iou = intersect / union

        return iou # shape (batch*num_sample), higher the better
