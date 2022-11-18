from typing import List, Tuple, Union, Dict, Any, Optional, cast
from seal.modules.loss import Loss
from seal.modules.loss.inference_net_loss import (
    MarginBasedLoss,
)
from allennlp.common.checks import ConfigurationError
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
import torch

# DVNLoss* are loss functions to train DVN,
# DVNScoreLoss* are loss functions to train infrence network with DVN.


class DVNLoss(Loss):
    """
    Loss function to train DVN, typically soft BCE loss.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        if self.score_nn is None:
            raise ConfigurationError("score_nn cannot be None for DVNLoss")

        if self.oracle_value_function is None:
            raise ConfigurationError(
                "oracle_value_function cannot be None for DVNLoss"
            )
        self.logging_buffer["predicted_score"] = LoggedScalarScalar()

    def compute_loss(
        self,
        predicted_score: torch.Tensor,  # (batch, num_samples)
        oracle_value: Optional[torch.Tensor],  # (batch, num_samples)
    ) -> torch.Tensor:
        raise NotImplementedError

    def _forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor],  # (batch, 1, ...)
        y_hat: torch.Tensor,  # (batch, num_samples, ...)
        y_hat_extra: Optional[torch.Tensor],  # (batch, num_samples)
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:

        predicted_score, oracle_value = self._get_values(
            x, labels, y_hat, y_hat_extra, buffer, **kwargs
        )
        self.log("predicted_score", predicted_score.detach().mean().item())

        return self.compute_loss(predicted_score, oracle_value)

    def _get_values(
        self,
        x: Any,
        labels: Optional[torch.Tensor],
        y_hat: torch.Tensor,
        y_hat_extra: Optional[torch.Tensor],
        buffer: Dict,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # labels shape (batch, 1, ...)
        # y_hat shape (batch, num_samples, ...)
        self.oracle_value_function = cast(
            OracleValueFunction, self.oracle_value_function
        )  # purely for typing, no runtime effect
        self.score_nn = cast(
            ScoreNN, self.score_nn
        )  # purely for typing, no runtime effect
        # score_nn always expects y to be normalized
        # do the normalization based on the task

        predicted_score = self.score_nn(
            x, y_hat, buffer, **kwargs
        )  # (batch, num_samples)

        if labels is not None:
            # For dvn we do not take gradient of oracle_score, so we detach y_hat
            oracle_score: Optional[torch.Tensor] = self.oracle_value_function(
                labels, y_hat.detach().clone(), **kwargs
            )  # (batch, num_samples)
        else:
            oracle_score = None

        return predicted_score, oracle_score


class DVNLossCostAugNet(Loss):
    """
    Loss function to train DVN, typically soft BCE loss.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        if self.score_nn is None:
            raise ConfigurationError("score_nn cannot be None for DVNLoss")

        if self.oracle_value_function is None:
            raise ConfigurationError(
                "oracle_value_function cannot be None for DVNLoss"
            )

    def compute_loss(
        self,
        predicted_score: torch.Tensor,  # (batch, num_samples)
        oracle_value: Optional[torch.Tensor],  # (batch, num_samples)
    ) -> torch.Tensor:
        raise NotImplementedError

    def _forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor],  # (batch, 1, ...)
        y_hat: torch.Tensor,  # (batch, num_samples, ...)
        y_hat_extra: Optional[torch.Tensor],  # (batch, num_samples)
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:

        predicted_score_list, oracle_value_list = self._get_values(
            x, labels, y_hat, y_hat_extra, buffer, **kwargs
        )

        loss = self.compute_loss(predicted_score_list[0], oracle_value_list[0])

        if predicted_score_list[1] is not None:
            loss += self.compute_loss(
                predicted_score_list[1], oracle_value_list[1]
            )

        return loss

    def _get_values(
        self,
        x: Any,
        labels: Optional[torch.Tensor],
        y_hat: torch.Tensor,
        y_hat_extra: Optional[torch.Tensor],
        buffer: Dict,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # labels shape (batch, 1, ...)
        # y_hat shape (batch, num_samples, ...)
        num_samples = y_hat[1]
        self.oracle_value_function = cast(
            OracleValueFunction, self.oracle_value_function
        )  # purely for typing, no runtime effect
        self.score_nn = cast(
            ScoreNN, self.score_nn
        )  # purely for typing, no runtime effect
        # score_nn always expects y to be normalized
        # do the normalization based on the task
        
        predicted_score = self.score_nn(
            x, y_hat, buffer, **kwargs
        )  # (batch, num_samples)

        if labels is not None:
            # For dvn we do not take gradient of oracle_score, so we detach y_hat
            oracle_score: Optional[torch.Tensor] = self.oracle_value_function(
                labels, y_hat.detach().clone(), **kwargs
            )  # (batch, num_samples)

            if y_hat_extra is not None:
                oracle_score_extra: Optional[
                    torch.Tensor
                ] = self.oracle_value_function(
                    labels, y_hat_extra.detach().clone(), **kwargs
                )  # (batch, num_samples)
            else:
                oracle_score_extra = None
        else:
            oracle_score = None
            oracle_score_extra = None

        if y_hat_extra is not None:
            predicted_score_extra = self.score_nn(
                x, y_hat_extra, buffer, **kwargs
            )  # (batch, num_samples)
        else:
            predicted_score_extra = None

        predicted_score_list = [predicted_score, predicted_score_extra]
        oracle_score_list = [oracle_score, oracle_score_extra]

        return predicted_score_list, oracle_score_list


class DVNScoreLoss(Loss):
    """
    Just uses score from the score network as the objective.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._inference_score_values = []
        if self.score_nn is None:
            raise ConfigurationError("score_nn cannot be None for DVNLoss")

    def compute_loss(
        self,
        predicted_score: torch.Tensor,  # (batch, num_samples)
    ) -> torch.Tensor:
        raise NotImplementedError

    def _forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor],  # (batch, 1, ...)
        y_hat: torch.Tensor,  # (batch, num_samples, ...)
        y_hat_extra: Optional[torch.Tensor],  # (batch, num_samples)
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:

        predicted_score = self._get_predicted_score(
            x, labels, y_hat, y_hat_extra, buffer, **kwargs
        )

        return self.compute_loss(predicted_score)

    def _get_predicted_score(
        self,
        x: Any,
        labels: Optional[torch.Tensor],
        y_hat: torch.Tensor,
        y_hat_extra: Optional[torch.Tensor],
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        # labels shape (batch, 1, ...)
        # y_hat shape (batch, num_samples, ...)
        self.score_nn = cast(
            ScoreNN, self.score_nn
        )  # purely for typing, no runtime effect

        # score_nn always expects y to be normalized
        # do the normalization based on the task
        
        predicted_score = self.score_nn(
            x, y_hat, buffer, **kwargs
        )  # (batch, num_samples)

        self._inference_score_values.append(float(torch.mean(predicted_score)))
        
        return predicted_score


class DVNScoreCostAugNet(Loss):
    """
    Just uses score from the score network as the objective,
    but also train CostAug network (i.e. Cost-Augmented Network).
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        if self.score_nn is None:
            raise ConfigurationError("score_nn cannot be None for DVNLoss")
        ## if I want to add "weight to the infnet vs. cost-augmented net"
        ## uncomment the following line and put it as hyperparamter.
        # self.inference_score_weight = inference_score_weight

    def compute_loss(
        self,
        predicted_score: torch.Tensor,  # (batch, num_samples)
    ) -> torch.Tensor:
        raise NotImplementedError

    def _forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor],  # (batch, 1, ...)
        y_hat: torch.Tensor,  # (batch, num_samples, ...)
        y_hat_extra: Optional[torch.Tensor],  # (batch, num_samples)
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:

        pred_score_infnet, pred_score_costaugnet = self._get_predicted_score(
            x, labels, y_hat, y_hat_extra, buffer, **kwargs
        )

        loss = self.compute_loss(pred_score_infnet)

        if pred_score_costaugnet is not None:
            loss += self.compute_loss(pred_score_costaugnet)

        return loss

    def _get_predicted_score(
        self,
        x: Any,
        labels: Optional[torch.Tensor],
        y_hat: torch.Tensor,
        y_hat_extra: Optional[torch.Tensor],
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        # labels shape (batch, 1, ...)
        # y_hat shape (batch, num_samples, ...)
        self.score_nn = cast(
            ScoreNN, self.score_nn
        )  # purely for typing, no runtime effect 

        predicted_score_infnet = self.score_nn(
            x, y_hat, buffer, **kwargs
        )  # (batch, num_samples)

        if y_hat_extra is not None:
            predicted_score_costaug = self.score_nn(
                x, y_hat_extra, buffer, **kwargs
            )  # (batch, num_samples)
        else:
            predicted_score_costaug = None

        return [predicted_score_infnet, predicted_score_costaug]


class DVNScoreAndCostAugLoss(MarginBasedLoss):
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
            oracle_cost
            + self.compute_loss(cost_augmented_inference_score)
            + self.inference_score_weight * self.compute_loss(inference_score)
            # the minus sign turns this into argmin objective
            # DVN scores are normalized by compute_loss of sigmoid.
        )

        return loss_unreduced
