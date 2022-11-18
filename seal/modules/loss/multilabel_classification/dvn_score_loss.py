from typing import List, Tuple, Union, Dict, Any, Optional
from seal.modules.loss import DVNScoreLoss, DVNScoreCostAugNet, DVNScoreAndCostAugLoss, Loss
import torch

def _normalize(y: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(y)


@Loss.register("multi-label-dvn-score")
class MultiLabelDVNScoreLoss(DVNScoreLoss):
    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        return _normalize(y)
        
    def compute_loss(
        self,
        predicted_score: torch.Tensor,  # logits of shape (batch, num_samples)
    ) -> torch.Tensor:
        return -torch.sigmoid(predicted_score)

@Loss.register("multi-label-score-loss")
class MultiLabelScoreLoss(DVNScoreLoss):
    """
    Non-DVN setup where score is not bounded in [0,1], 
    however the only thing we need is score from scoreNN, 
    so it's better to share with DVNScoreLoss.
    """
    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        return _normalize(y)
        
    def compute_loss(
        self,
        predicted_score: torch.Tensor,  # logits of shape (batch, num_samples)
    ) -> torch.Tensor:
        return -predicted_score


@Loss.register("multi-label-dvn-ca-score")
class MultiLabelDVNScoreCA(DVNScoreCostAugNet):
    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        return _normalize(y)
        
    def compute_loss(
        self,
        predicted_score: torch.Tensor,  # logits of shape (batch, num_samples)
    ) -> torch.Tensor:
        return -torch.sigmoid(predicted_score)



@Loss.register("multi-label-dvn-plus-ca-loss")
class MultiLabelDVNScorePlusCostAugLoss(DVNScoreAndCostAugLoss):
    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        return _normalize(y)

    def compute_loss(
        self,
        predicted_score: torch.Tensor,  # logits of shape (batch, num_samples)
    ) -> torch.Tensor:
        # there is already minus sign on DVNScoreAndCostAugLoss
        return torch.sigmoid(predicted_score)
        

@Loss.register("multi-label-unnorm-dvn-plus-ca-loss")
class MultiLabelDVNUnNormScorePlusCostAugLoss(DVNScoreAndCostAugLoss):
    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        return _normalize(y)

    def compute_loss(
        self,
        predicted_score: torch.Tensor,  # logits of shape (batch, num_samples)
    ) -> torch.Tensor:
        # there is already minus sign on DVNScoreAndCostAugLoss
        return predicted_score
        
