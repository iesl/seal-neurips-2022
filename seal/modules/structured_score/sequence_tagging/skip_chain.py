from typing import Any
from seal.modules.structured_score.sequence_tagging.linear_chain import LinearChain
from seal.modules.structured_score.structured_score import StructuredScore
import torch
import torch.nn as nn
import numpy as np


@StructuredScore.register("skip-chain")
class SkipChain(LinearChain):
    def __init__(self, num_tags: int, M: int, **kwargs: Any):
        """
        TODO: Change kwargs to take hidden size and output size
        """
        super().__init__(num_tags)
        self.M = M
        self.W = nn.Parameter(
            torch.FloatTensor(np.random.uniform(-0.02, 0.02, (self.M, num_tags + 1, num_tags + 1)).astype('float32')))
