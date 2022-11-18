from typing import List, Tuple, Union, Dict, Any, Optional
from seal.modules.loss import (
    DVNLossCostAugNet,
    DVNLoss,
    Loss,
)
import torch

def _normalize(y: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(y)

@Loss.register("multi-label-dvn-bce")
class MultiLabelDVNCrossEntropyLoss(DVNLoss):
    
    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        return _normalize(y)

    def compute_loss(
        self,
        predicted_score: torch.Tensor,  # logits of shape (batch, num_samples)
        oracle_value: Optional[torch.Tensor],  # (batch, num_samples)
    ) -> torch.Tensor:
        # both oracle values and predicted scores are higher the better
        # The oracle value will be something like f1 or 1-hamming_loss
        # which will take values in [0,1] with best value being 1.
        # Predicted score are logits, hence bce with logit will
        # internally map them to [0,1]

        return torch.nn.functional.binary_cross_entropy_with_logits(
            predicted_score,
            oracle_value,
            reduction=self.reduction,
        )


@Loss.register("multi-label-dvn-ca-bce")
class MultiLabelDVNCostAugCrossEntropyLoss(DVNLossCostAugNet):

    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        return _normalize(y)

    def compute_loss(
        self,
        predicted_score: torch.Tensor,  # logits of shape (batch, num_samples)
        oracle_value: Optional[torch.Tensor],  # (batch, num_samples)
    ) -> torch.Tensor:
        # both oracle values and predicted scores are higher the better
        # The oracle value will be something like f1 or 1-hamming_loss
        # which will take values in [0,1] with best value being 1.
        # Predicted score are logits, hence bce with logit will
        # internally map them to [0,1]

        return torch.nn.functional.binary_cross_entropy_with_logits(
            predicted_score,
            oracle_value,
            reduction=self.reduction,
        )


@Loss.register("zero-dvn-loss")
class ZeroLoss(MultiLabelDVNCrossEntropyLoss):
    """
    Loss function to give zero signal to DVN
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs) 
                
    def forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor],  # (batch, 1, ...)
        y_hat: torch.Tensor,  # (batch, num_samples, ...)
        y_hat_extra: Optional[torch.Tensor],  # (batch, num_samples)
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        return 0 * super().forward(
            x, labels, y_hat, y_hat_extra, buffer, **kwargs
        )
