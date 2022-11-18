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
import torch

# Losses to train score-NN with noise contrastive estimation (NCE) techniques


class NCELoss(Loss):
    """
    Computes logits = s(x, y) - distance(y, y_hat),
    where y = cat([label], samples), and y_hat is the probability of sampler.
    """

    def __init__(self, num_samples: int = 10, **kwargs: Any):
        super().__init__(**kwargs)
        self.num_samples = num_samples

        if self.score_nn is None:
            raise ConfigurationError(
                "score_nn cannot be None for NCERankingLoss"
            )

    def sample(
        self,
        probs: torch.Tensor,  # (batch, 1, ...)
    ) -> torch.Tensor:  # (batch, num_samples, ...)
        """
        Continuous or discrete sampling.

        Need to override this method to implement different kinds of sampling strategies
        as well as for different kinds of tasks.
        """
        raise NotImplementedError

    def distance(
        self,
        samples: torch.Tensor,  # (batch, num_samples, ...)
        probs: torch.Tensor,  # (batch, num_samples, ...)
    ) -> torch.Tensor:  # (batch, num_samples)
        """
        This is the generalization of Pn. It is some form of distance between
        samples and probs.

        Need to override this for different tasks and different signs of Pn
        """
        raise NotImplementedError

    def compute_loss(
        self,
        x: torch.Tensor,
        y_hat: torch.Tensor,  # (batch, 1, ...) normalized
        labels: torch.Tensor,  # (batch, 1, ...)
        buffer: Dict,
    ) -> torch.Tensor:
        """
        The function that
        """
        raise NotImplementedError

    def _forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor],  # (batch, 1, ...)
        y_hat: torch.Tensor,  # (batch, 1, ...) normalized
        y_hat_extra: Optional[torch.Tensor],  # (batch, 1, ...)
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        assert y_hat.shape[1] == 1
        assert labels is not None
        loss_unreduced = self.compute_loss(
            x, y_hat, labels, buffer
        )  # (batch, num_samples)

        return loss_unreduced


class NCERankingLoss(NCELoss):
    def __init__(self, use_scorenn: bool = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.use_scorenn = use_scorenn
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")

    def compute_loss(
        self,
        x: torch.Tensor,
        y_hat: torch.Tensor,  # (batch, 1, ...)
        labels: torch.Tensor,  # (batch, 1, ...)
        buffer: Dict,
    ) -> torch.Tensor:
        samples = self.sample(y_hat).to(
            dtype=y_hat.dtype
        )  # (batch, num_samples, ...)
        y = torch.cat(
            (labels.to(dtype=samples.dtype), samples), dim=1
        )  # (batch, 1+num_samples, ...)
        distance = self.distance(
            y, y_hat.expand_as(y)
        )  # (batch, 1+num_samples) # does the job of Pn
        if self.use_scorenn:
            score = self.score_nn(
                x, y, buffer
            )  # type:ignore # (batch, 1+num_samples)
            assert not distance.requires_grad
        else:
            score = 0 
                    
        new_score = score - distance  # (batch, 1+num_samples)
        ranking_loss = self.cross_entropy(
            new_score,
            torch.zeros(
                new_score.shape[0], dtype=torch.long, device=new_score.device
            ),  # (batch,)
        )

        return ranking_loss
