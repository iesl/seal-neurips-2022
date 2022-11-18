from typing import List, Tuple, Union, Dict, Any, Optional
import torch
from seal.modules.oracle_value_function import (
    OracleValueFunction,
)


def compute(
    labels: torch.Tensor,  #: (batch*num_samples, ...)
    y_hat: torch.Tensor,  #: (batch*num_samples, ...)
) -> torch.Tensor:

    intersection = torch.sum(
        torch.minimum(y_hat, labels), dim=-1
    )  # (batch*num_samples,)
    union = torch.sum(torch.maximum(y_hat, labels), dim=-1)

    return 2.0 * intersection / (intersection + union)


@OracleValueFunction.register("per-instance-f1")
class PerInstanceF1(OracleValueFunction):
    """
    Return oracle value that is based on per instance f1 score.

    Single f1 score will be in [0,1] with 1 being the best, we return f1
    as is for the oracle value.
    """

    @property
    def upper_bound(self) -> float:
        return 1.0

    def compute(
        self,
        labels: torch.Tensor,  #: (batch*num_samples, ...)
        y_hat: torch.Tensor,  #: (batch*num_samples, ...)
        mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return compute(labels, y_hat)


@OracleValueFunction.register("per-instance-lsef1")
class PerInstanceLSEF1(OracleValueFunction):
    """
    Return oracle value that is based on per instance f1 score that is
    computed by replacing min and max operations with logsumexp.

    Sing f1 score will be in [0,1] with 1 being the best, we return f1
    as is for the oracle value.
    """

    @property
    def upper_bound(self) -> float:
        raise NotImplementedError

    def compute(
        self,
        labels: torch.Tensor,  #: (batch*num_samples, ...)
        y_hat: torch.Tensor,  #: (batch*num_samples, ...)
        mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:

        intersection = torch.sum(
            -0.0001 * torch.logaddexp(-y_hat / 0.0001, -labels / 0.0001),
            dim=-1,
        )  # (batch*num_samples,)
        union = torch.sum(
            0.0001 * torch.logaddexp(y_hat / 0.0001, labels / 0.0001), dim=-1
        )

        return 2.0 * intersection / (intersection + union)
