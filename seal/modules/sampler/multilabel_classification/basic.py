from typing import List, Tuple, Union, Dict, Any, Optional, overload
from seal.modules.sampler import (
    Sampler,
    BasicSampler,
)
import torch
from seal.modules.score_nn import ScoreNN
from seal.modules.oracle_value_function import (
    OracleValueFunction,
)
from seal.modules.loss import Loss

from seal.modules.multilabel_classification_task_nn import (
    MultilabelTaskNN,
)


@Sampler.register("multi-label-basic")
class MultilabelClassificationBasicSampler(BasicSampler):
    @property
    def is_normalized(self) -> bool:
        return True

    @overload
    def normalize(self, y: None) -> None:
        ...

    @overload
    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        ...

    def normalize(self, y: Optional[torch.Tensor]) -> Optional[torch.Tensor]:

        if y is not None:
            return torch.sigmoid(y)
        else:
            return None

    @property
    def different_training_and_eval(self) -> bool:
        return False
