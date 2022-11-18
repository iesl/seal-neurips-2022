from typing import Any, Optional

import torch

from seal.modules.oracle_value_function import (
    OracleValueFunction,
)


@OracleValueFunction.register("manhattan")
class ManhattanDistanceValueFunction(OracleValueFunction):
    """Return oracle value that is based on manhattan distance (L1 Norm), specifically
    negative of L1 norm.

    This value function is bounded from above by 0.
    """

    @property
    def upper_bound(self) -> float:
        return 0.0

    def compute(
        self,
        labels: torch.Tensor,
        y_hat: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        distance = torch.mean(torch.abs(labels - y_hat), dim=-1)

        if mask is not None:
            if mask.dim() == 3:
                # we might have added an extra dim to mask
                mask = mask.squeeze(1)
            distance *= mask
            distance = torch.mean(distance, dim=-1)

        return -distance  # this value is higher the better so we flip the sign
