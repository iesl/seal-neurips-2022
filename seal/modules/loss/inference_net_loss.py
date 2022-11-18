from typing import Any, Optional, Tuple, cast, Dict, Callable

import numpy as np
import torch
from allennlp.common.checks import ConfigurationError

from seal.modules.loss import Loss
from seal.modules.oracle_value_function import (
    OracleValueFunction,
)
from seal.modules.score_nn import ScoreNN
from seal.modules.logging import (
    LoggingMixin,
    LoggedValue,
    LoggedScalarScalar,
    LoggedScalarScalarSample,
    LoggedNPArrayNPArraySample,
)


class MarginBasedLoss(Loss):
    """
    Implements the losses described in the SPEN+Inference Network papers.

    We compute $Delta(y_hat, y^*)$ as (oracle_score(y^*) - oracle_score(y_hat)).

    1. Margin Rescaled Loss:

    """

    margin_types: Dict[
        str,
        Callable[
            [torch.Tensor, float, torch.Tensor, torch.Tensor],
            torch.Tensor,
        ],
    ] = {
        "margin-rescaled-zero-truncation": (
            lambda oracle_cost, oracle_cost_weight, cost_augmented_inference_score, ground_truth_score: torch.relu(
                oracle_cost * (1 / oracle_cost_weight)
                - (ground_truth_score - cost_augmented_inference_score)
            )
        ),
        "slack-rescaled-zero-truncation": (
            lambda oracle_cost, oracle_cost_weight, cost_augmented_inference_score, ground_truth_score: (
                oracle_cost
                * (1 / oracle_cost_weight)
                * torch.relu(
                    1.0 - (ground_truth_score - cost_augmented_inference_score)
                )
            )
        ),
        "perceptron-zero-truncation": (
            lambda oracle_cost, oracle_cost_weight, cost_augmented_inference_score, ground_truth_score: torch.relu(
                cost_augmented_inference_score - ground_truth_score
            )
        ),
        "contrastive-zero-truncation": (
            lambda oracle_cost, oracle_cost_weight, cost_augmented_inference_score, ground_truth_score: torch.relu(
                1.0 - (ground_truth_score - cost_augmented_inference_score)
            )
        ),
    }

    def __init__(
        self,
        score_nn: ScoreNN,
        oracle_value_function: OracleValueFunction,
        oracle_cost_weight: float = 1.0,
        reduction: Optional[str] = "none",
        normalize_y: bool = True,
        margin_type: str = "margin-rescaled-zero-truncation",
        perceptron_loss_weight: float = 0.0,
        **kwargs: Any,
    ):
        """
        margin_type: one of [''] as described here https://arxiv.org/pdf/1803.03376.pdf
        cross_entropy: set True when training inference network and false for energy function, default True
        zero_truncation: set True when training for energy function, default False

        """
        super().__init__(
            score_nn=score_nn,
            oracle_value_function=oracle_value_function,
            reduction=reduction,
            normalize_y=normalize_y,
        )

        if self.score_nn is None:
            raise ConfigurationError("score_nn cannot be None")

        if self.oracle_value_function is None:
            raise ConfigurationError("oracle_value_function cannot be None")

        if margin_type not in self.margin_types:
            raise ConfigurationError(
                f"margin_type must be one of {self.margin_types}"
            )

        if oracle_cost_weight == 0:
            raise ConfigurationError("oracle_cost_weight must be non zero")

        self.oracle_cost_weight = oracle_cost_weight
        self.margin_type = margin_type
        self.perceptron_loss_weight = perceptron_loss_weight
        # setup logging.

        if not self.log_key:
            self.log_key = "margin_loss"
        self.logging_buffer["oracle_cost"] = LoggedScalarScalar()
        self.logging_buffer["ca_score"] = LoggedScalarScalar()
        self.logging_buffer["inf_score"] = LoggedScalarScalar()
        self.logging_buffer["gt_score"] = LoggedScalarScalar()

    def _forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor],  # (batch, 1, ...)
        y_hat: torch.Tensor,  # (batch, num_samples, ...) unnormalized
        y_hat_extra: Optional[
            torch.Tensor
        ],  # (batch, num_samples, ...), unnormalized
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        y_inf = y_hat
        y_cost_aug = y_hat_extra
        assert buffer is not None
        (
            oracle_cost,
            cost_aug_score,
            inference_score,
            ground_truth_score,
        ) = self._get_values(x, labels, y_inf, y_cost_aug, buffer)
        loss_unreduced = self.margin_types[self.margin_type](
            oracle_cost,
            self.oracle_cost_weight,
            cost_aug_score,
            ground_truth_score,
        )

        if self.perceptron_loss_weight:
            loss_unreduced = loss_unreduced + self.perceptron_loss_weight * (
                torch.relu(inference_score - ground_truth_score)
            )

        return loss_unreduced

    def _get_values(
        self,
        x: Any,
        labels: Optional[torch.Tensor],
        y_hat: torch.Tensor,  # Assumed to be unnormalized
        y_cost_aug: Optional[torch.Tensor],  # assumed to be unnormalized
        buffer: dict,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # labels shape (batch, 1, ...)
        # y_hat shape (batch, num_samples, ...)
        self.oracle_value_function = cast(
            OracleValueFunction, self.oracle_value_function
        )  # purely for typing, no runtime effect
        self.score_nn = cast(
            ScoreNN, self.score_nn
        )  # purely for typing, no runtime effect
        assert (
            labels is not None
        )  # if you call this loss, labels cannot be None

        ground_truth_score = self.score_nn(
            x, labels.to(dtype=y_hat.dtype), buffer
        )
        inference_score = self.score_nn(x, y_hat, buffer)

        if y_cost_aug is None:
            y_cost_aug = y_hat
            cost_aug_score = torch.zeros_like(inference_score)
        else:
            cost_aug_score = self.score_nn(x, y_cost_aug, buffer)

        oracle_cost: torch.Tensor = self.oracle_value_function.compute_as_cost(
            labels, y_cost_aug, mask=buffer.get("mask")
        )  # (batch, num_samples)
        # log
        self.log("oracle_cost", oracle_cost.detach().mean().item())
        self.log("ca_score", cost_aug_score.detach().mean().item())
        self.log("inf_score", inference_score.detach().mean().item())
        self.log("gt_score", ground_truth_score.detach().mean().item())

        return (
            oracle_cost,
            cost_aug_score,
            inference_score,
            ground_truth_score,
        )


class InferenceLoss(MarginBasedLoss):
    """
    The class exclusively outputs loss (lower the better) to train the paramters of the inference net.
    Last equation in the section 2.3 in "An Exploration of Arbitrary-Order Sequence Labeling via Energy-Based Inference Networks".

    Note:
        Right now we always drop zero truncation.
    """

    def __init__(self, inference_score_weight: float, **kwargs: Any):
        super().__init__(**kwargs)
        self.inference_score_weight = inference_score_weight

    def _forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor],  # (batch, 1, ...)
        y_hat: torch.Tensor,  # (batch, num_samples, ...) might be unnormalized
        y_hat_extra: Optional[
            torch.Tensor
        ],  # (batch, num_samples, ...), might be unnormalized
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        y_inf = y_hat
        y_cost_aug = y_hat_extra
        assert buffer is not None
        (
            oracle_cost,
            cost_augmented_inference_score,
            inference_score,
            ground_truth_score,
        ) = self._get_values(x, labels, y_inf, y_cost_aug, buffer)
        loss_unreduced = -(
            oracle_cost * (1 / self.oracle_cost_weight)
            + cost_augmented_inference_score
            + self.inference_score_weight * inference_score
        )  # the minus sign turns this into argmin objective

        return loss_unreduced


class StructuredSVMLoss(MarginBasedLoss):
    """
        The class outputs structured SVM (SSVM) Loss
        Equation 8 in "Structured Prediction Energy Networks".
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def _forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor],  # (batch, 1, ...)
        y_hat: torch.Tensor,  # (batch, num_samples, ...) might be unnormalized
        y_hat_extra: Optional[
            torch.Tensor
        ],  # (batch, num_samples, ...), might be unnormalized
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        y_inf = y_hat
        y_cost_aug = y_hat_extra
        assert buffer is not None
        (
            oracle_cost,
            cost_augmented_inference_score,
            inference_score,
            ground_truth_score,
        ) = self._get_values(x, labels, y_inf, y_cost_aug, buffer)

        loss_unreduced = torch.relu(
                oracle_cost * (1 / self.oracle_cost_weight)
                - (ground_truth_score - inference_score)
        )

        return loss_unreduced
