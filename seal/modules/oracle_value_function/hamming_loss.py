from typing import List, Tuple, Union, Dict, Any, Optional
import torch
from seal.modules.oracle_value_function import (
    OracleValueFunction,
)


@OracleValueFunction.register("hamming")
class HammingValueFunction(OracleValueFunction):
    """
    Return oracle value that is based on normalized hamming loss.

    Since, normalized hamming loss takes values in [0,1] with 1 being the worst.
    We return hamming_value = (1 - hamming_loss) which takes value in [0, 1] with
    1 being the best.
    """

    @property
    def upper_bound(self) -> float:
        return 1.0

    def compute(
        self,
        labels: torch.Tensor,  #: (batch*num_samples, ...)
        y_hat: torch.Tensor,  #: (batch*num_samples, ...)
        **kwargs: Any,
    ) -> torch.Tensor:
        return 1.0 - torch.mean(
            torch.abs(labels - y_hat), dim=-1
        )  # (batch*num_samples,) higher the better with values in [0,1]
