from typing import List, Tuple, Union, Dict, Any, Optional
from seal.modules.structured_score.structured_score import StructuredScore
import torch


@StructuredScore.register("fully_connected")
class FullyConnected(StructuredScore):
    def __init__(self, **kwargs: Any):
        """
        TODO: Change kwargs to take hidden size and output size
        """
        super().__init__()
        # TODO: initialize weights
        # self.W =

    def forward(
        self,
        y: torch.Tensor,
        mask: torch.BoolTensor = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        # implement
        pass
