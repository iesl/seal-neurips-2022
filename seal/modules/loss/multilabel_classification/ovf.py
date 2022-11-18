from typing import List, Tuple, Union, Dict, Any, Optional
import torch
from seal.modules.loss import Loss


@Loss.register("multi-label-ovf")
class MultiLabelOVFLoss(Loss):
    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(y)

    def _forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor],  # (batch, 1, num_labels)
        y_hat: torch.Tensor,  # (batch, 1, num_labels)
        y_hat_extra: Optional[torch.Tensor],
        buffer: Optional[Dict] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        assert labels is not None
        assert self.oracle_value_function is not None
        
        return -self.oracle_value_function(labels, y_hat)  # (batch,)
