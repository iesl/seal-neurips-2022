from typing import Any, Optional, Tuple, cast, Union, Dict

import torch
from allennlp.common.checks import ConfigurationError
from allennlp.nn import util
from torch.nn.functional import relu
import torch.nn.functional as F

from seal.modules.loss import Loss, DVNScoreLoss
from seal.modules.oracle_value_function import (
    OracleValueFunction,
)
from seal.modules.score_nn import ScoreNN

from seal.modules.loss.inference_net_loss import (
    MarginBasedLoss,
    InferenceLoss,
)


def _normalize(y: torch.Tensor) -> torch.Tensor:
    return torch.softmax(y, dim=-1)


@Loss.register("sequence-tagging-margin-based")
class SequenceTaggingMarginBasedLoss(MarginBasedLoss):
    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        return _normalize(y)


@Loss.register("sequence-tagging-inference")
class SequenceTaggingInferenceLoss(InferenceLoss):
    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        return _normalize(y)

@Loss.register("sequence-tagging-score-loss")
class SequenceTaggingScoreLoss(DVNScoreLoss):
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