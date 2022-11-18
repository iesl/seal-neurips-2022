from typing import Any, Optional, Dict
import torch
from allennlp.nn import util
import torch.nn.functional as F
from seal.modules.loss import Loss, DVNScoreLoss

@Loss.register("weizmann-horse-seg-bce")
class WeizmannHorseSegBCELoss(Loss):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
        self._loss = 0.
        self._count = 0.

    def _forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor], # (b, c=1, h, w)
        y_hat: torch.Tensor, # (b, c=1, h, w)
        y_hat_extra: Optional[torch.Tensor] = None,
        buffer: Dict = None,
        **kwargs: Any,
    ) -> torch.Tensor:

        y_hat = y_hat.view(*y_hat.size()[:-2], -1).squeeze(-3) # (b, h*w)
        labels = labels.view(*labels.size()[:-2], -1).squeeze(-3) # (b, h*w)
        loss = self.loss_fn(y_hat, labels.to(dtype=y_hat.dtype)).mean(dim=-1) # (b,)

        self._loss += loss
        self._count += 1
        return loss

    def get_metrics(self, reset: bool = False):
        metrics = {}
        if self._loss:
            metrics = {"cross_entropy_loss": self._loss / self._count}
        if reset:
            self._loss = 0.
            self._count = 0.
        return metrics


@Loss.register("weizmann-horse-seg-ce")
class WeizmannHorseSegCELoss(Loss):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        self._loss = 0.
        self._count = 0.

    def _forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor], # (b, c=1, h, w)
        y_hat: torch.Tensor, # (b, c=2, h, w)
        y_hat_extra: Optional[torch.Tensor] = None,
        buffer: Dict = None,
        **kwargs: Any,
    ) -> torch.Tensor:

        loss = self.loss_fn(y_hat, labels.squeeze(-3).long()) # (b, h, w)
        loss = loss.view(-1, loss.size()[-2]*loss.size()[-1]).mean(dim=-1) # (b,)

        self._loss += loss
        self._count += 1

        return loss

    def get_metrics(self, reset: bool = False):
        metrics = {}
        if self._loss:
            metrics = {"cross_entropy_loss": self._loss / self._count}
        if reset:
            self._loss = 0.
            self._count = 0.
        return metrics


@Loss.register("weizmann-horse-seg-dvn-score-loss")
class WeizmannHorseSegDVNScoreLoss(DVNScoreLoss):
    """
    Non-DVN setup where score is not bounded in [0,1],
    however the only thing we need is score from scoreNN,
    so it's better to share with DVNScoreLoss.
    """

    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        if y.size()[-3] == 1:  # (b, c=1, h, w)
            return torch.sigmoid(y)
        elif y.size()[-3] == 2:  # (b, c=2, h, w)
            return torch.softmax(y, dim=-3)[..., 1, :, :].unsqueeze(-3) # only the horse channel
        else:
            raise

    def compute_loss(
            self,
            predicted_score: torch.Tensor,  # (b, n)
    ) -> torch.Tensor:
        return -torch.sigmoid(predicted_score)


@Loss.register("weizmann-horse-seg-score-loss")
class WeizmannHorseSegScoreLoss(DVNScoreLoss):
    """
    Non-DVN setup where score is not bounded in [0,1],
    however the only thing we need is score from scoreNN,
    so it's better to share with DVNScoreLoss.
    """

    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        if y.size()[-3] == 1:  # (b, c=1, h, w)
            return torch.sigmoid(y)
        elif y.size()[-3] == 2:  # (b, c=2, h, w)
            return torch.softmax(y, dim=-3)[..., 1, :, :].unsqueeze(-3) # only the horse channel
        else:
            raise

    def compute_loss(
            self,
            predicted_score: torch.Tensor,  # (b, n)
    ) -> torch.Tensor:
        return -predicted_score