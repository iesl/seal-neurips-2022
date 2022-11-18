from typing import List, Tuple, Union, Dict, Any, Optional
from enum import Enum
import torch


class ModelMode(Enum):
    UPDATE_TASK_NN: str = "task_nn"
    UPDATE_SCORE_NN: str = "score_nn"
    COMPUTE_SCORE: str = "compute_score_nn"

    def flip(self) -> "ModelMode":
        if self == ModelMode.UPDATE_TASK_NN:
            return ModelMode.UPDATE_SCORE_NN
        elif self == ModelMode.UPDATE_SCORE_NN:
            return ModelMode.UPDATE_TASK_NN
        else:
            raise RuntimeError(f"{self} cannot be flipped.")

    def mark_parameter_with_model_mode(
        self, param: torch.nn.Parameter
    ) -> None:
        param.model_mode = self

    def is_parameter_model_mode(self, param: torch.nn.Parameter) -> bool:
        if hasattr(param, "model_mode"):
            return param.model_mode == self
        else:
            return False

    @classmethod
    def hasattr_model_mode(cls, obj: Any) -> bool:
        return hasattr(obj, "model_mode")

    @classmethod
    def getattr_model_mode(cls, obj: Any) -> "ModelMode":
        return getattr(obj, "model_mode")
