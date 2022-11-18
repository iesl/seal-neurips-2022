import logging
from typing import Any, Optional, Dict, List, Tuple
from overrides import overrides
import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from seal.metrics import SegIoU
from seal.modules.loss import Loss
from seal.modules.sampler import Sampler
from .base import ScoreBasedLearningModel

logger = logging.getLogger(__name__)


@Model.register("weizmann-horse-seg", constructor="from_partial_objects")
@Model.register("seal-weizmann-horse-seg", constructor="from_partial_objects_with_shared_tasknn")
class WeizmannHorseSegModel(ScoreBasedLearningModel):

    def __init__(
        self,
        vocab: Vocabulary,
        sampler: Sampler,
        loss_fn: Loss,
        **kwargs: Any,
    ):
        super().__init__(vocab, sampler, loss_fn, **kwargs)
        self.instantiate_metrics()

    def instantiate_metrics(self) -> None:
        self._seg_iou = SegIoU()

    @overrides
    def unsqueeze_labels(self, labels: torch.Tensor) -> torch.Tensor:
        return labels.unsqueeze(-4) # (b, n=1, c=1, h, w)

    @overrides
    def squeeze_y(self, y: torch.Tensor) -> torch.Tensor:
        return y

    @overrides
    def construct_args_for_forward(self, **kwargs: Any) -> Dict:
        _forward_args = {}
        _forward_args["buffer"] = self.initialize_buffer(**kwargs)
        _forward_args["x"] = kwargs.pop("image")
        _forward_args["labels"] = kwargs.pop("mask")
        return {**_forward_args, **kwargs}

    @overrides
    def calculate_metrics(
        self,
        x: Any,
        labels: torch.Tensor, # (b, )
        y_hat: torch.Tensor,
        buffer: Dict,
        results: Dict,
        **kwargs: Any,
    ) -> None:
        self._seg_iou(y_hat.detach(), labels.long())

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"seg_iou": self._seg_iou.get_metric(reset=reset)}

