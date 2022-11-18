from typing import List, Tuple, Union, Dict, Any, Optional
from allennlp.common.registrable import Registrable
import torch
from seal.modules.score_nn import ScoreNN
from seal.modules.oracle_value_function import (
    OracleValueFunction,
)
from seal.modules.logging import (
    LoggingMixin,
    LoggedValue,
    LoggedScalarScalar,
    LoggedScalarScalarSample,
    LoggedNPArrayNPArraySample,
)
from allennlp.common.lazy import Lazy


class Loss(LoggingMixin, torch.nn.Module, Registrable):
    """Base class for all the loss functions.

    In some cases, this will only act as a wrapper around loss modules from pytorch.
    """

    default_implementation = "zero"
    allowed_reductions = ["sum", "mean", "none"]

    def __init__(
        self,
        score_nn: Optional[ScoreNN] = None,
        oracle_value_function: Optional[OracleValueFunction] = None,
        reduction: Optional[str] = "sum",
        normalize_y: bool = False,
        **kwargs: Any,
    ):
        """
        Args:
            score_nn: Needed if we are train a SPEN or DVN and need to compute v(x,y), i.e. the value of an input-output instance.
            oracle_value_function: Needed if we are doing DVN or SPEN.
            normalize_y: y_hat and y_hat_extra might not always be normalized. Set this flag to True in such cases to inform the loss.
        """
        super().__init__(**kwargs)  # type: ignore
        self.score_nn = score_nn
        self.oracle_value_function = oracle_value_function

        if reduction not in self.allowed_reductions:
            raise ValueError(
                f"reduction should be one of {self.allowed_reductions}"
            )
        self.reduction = reduction
        self.normalize_y = normalize_y
        # we will use empty string to log the main loss value
        self.logging_buffer[""] = LoggedScalarScalar()

    def forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor],  #: shape (batch, 1, ...)
        y_hat: torch.Tensor,  #: shape (batch, num_samples, ...), assumed to be unnormalized logits
        y_hat_extra: Optional[
            torch.Tensor
        ],  # y_hat_probabilities or y_hat_cost_augmented, assumed to be unnormalized logits
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:

        if self.normalize_y:
            y_hat = self.normalize(y_hat)
            if y_hat_extra is not None:
                y_hat_extra = self.normalize(y_hat_extra)

        loss_unreduced = self._forward(
            x, labels, y_hat, y_hat_extra, buffer, **kwargs
        )

        loss = self.reduce(loss_unreduced)
        self.log("", loss.detach().mean().item())

        return loss

    def reduce(self, loss_unreduced: torch.Tensor) -> torch.Tensor:
        if self.reduction == "sum":
            return torch.sum(loss_unreduced)
        elif self.reduction == "mean":
            return torch.mean(loss_unreduced)
        elif self.reduction == "none":
            return loss_unreduced
        else:
            raise ValueError

    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        """ To normalize y based on the task."""
        raise NotImplementedError

    def _forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor],  #: shape (batch, 1, ...)
        y_hat: torch.Tensor,  #: shape (batch, num_samples, ...)
        y_hat_extra: Optional[
            torch.Tensor
        ],  # y_hat_probabilities or y_hat_cost_augmented
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Args:
            x: Any input tensor (MLC task) or dictionary of tensors (sequence tagging task) based on the task.
        Returns:
            loss of shape
                1. (batch, num_samples) if reduction is None
                2. (,), ie a scaler loss if reduction is sum or mean
        """
        raise NotImplementedError


@Loss.register("zero")
class ZeroLoss(Loss):
    """
    Loss function which always outputs zero.
    """

    def __init__(self, reduction: str = "none", **kwargs: Any):
        reduction = "none"
        super().__init__(reduction=reduction, **kwargs)

    def forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor],  # (batch, 1, ...)
        y_hat: torch.Tensor,  # (batch, num_samples, ...)
        y_hat_extra: Optional[torch.Tensor],  # (batch, num_samples)
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        return torch.zeros(1, device=y_hat.device).sum()


@Loss.register("combination-loss")
class CombinationLoss(Loss):
    def __init__(
        self,
        constituent_losses: List[Loss],
        loss_weights: Optional[List[float]] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        assert len(constituent_losses) > 0, "Cannot have empty list of losses"
        self.constituent_losses = torch.nn.ModuleList(constituent_losses)
        self.loss_weights = loss_weights or [1.0] * len(
            self.constituent_losses
        )
        assert len(self.loss_weights) == len(self.constituent_losses)
        # register children losses

        for cl in self.constituent_losses:
            self.logging_children.append(cl)

    def _forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor],  #: shape (batch, 1, ...)
        y_hat: torch.Tensor,  #: shape (batch, num_samples, ...)
        y_hat_extra: Optional[
            torch.Tensor
        ],  # y_hat_probabilities or y_hat_cost_augmented
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        losses = [
            l_(x, labels, y_hat, y_hat_extra, buffer, **kwargs)
            for l_ in self.constituent_losses
        ]
        total_loss = self.loss_weights[0] * losses[0]

        for w, l in zip(self.loss_weights[1:], losses[1:]):
            total_loss = total_loss + w * l

        return total_loss


@Loss.register("negative")
class NegativeLoss(Loss):
    """Flips the sign of the loss"""

    def __init__(
        self,
        constituent_loss: Loss,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.constituent_loss = constituent_loss
        # register child for logging
        self.logging_children.append(self.constituent_loss)

    def _forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor],  #: shape (batch, 1, ...)
        y_hat: torch.Tensor,  #: shape (batch, num_samples, ...)
        y_hat_extra: Optional[
            torch.Tensor
        ],  # y_hat_probabilities or y_hat_cost_augmented
        buffer: Dict = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return -self.constituent_loss(
            x, labels, y_hat, y_hat_extra, buffer, **kwargs
        )
