from typing import List, Tuple, Union, Dict, Any, Optional
from .score_nn import ScoreNN
import torch
from allennlp.data import TextFieldTensors, Vocabulary
import allennlp.nn.util as util


@ScoreNN.register("sequence-tagging")
class SequenceTaggingScoreNN(ScoreNN):
    def compute_local_score(  # type:ignore
        self,
        x: TextFieldTensors,
        y: torch.Tensor,  #: shape (batch, num_samples or 1, seq_len, num_tags)
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Args:
            y: tensor of labels of shape (batch, seq_len, tags)
        """
        # y_local = buffer.get("y_local")

        # if y_local is None:
        #     y_local = self.task_nn(
        #         x, buffer
        #     )  # (batch, ...) of unormalized logits
        #     buffer["y_local"] = y_local

        y_local = self.task_nn(
            x, buffer
        )  # (batch, ...) of unormalized logits

        mask = buffer.get("mask")
        mask = mask.unsqueeze(1)
        if mask is None:
            mask = util.get_text_field_mask(x)
            mask = mask.unsqueeze(dim=1)
            buffer["mask"] = mask

        local_score = torch.sum(
            y_local.unsqueeze(1) * y, dim=-1
        )  # (batch, num_samples, seq)
        local_score = torch.sum(local_score * mask, dim=-1)

        return local_score  # (batch, num_samples)

    def forward(
        self,
        x: TextFieldTensors,
        y: torch.Tensor,
        buffer: Dict,
        **kwargs: Any,
    ) -> Optional[torch.Tensor]:
        score = None

        local_score = self.compute_local_score(x, y, buffer=buffer)

        global_score = self.compute_global_score(y, buffer)

        return local_score + global_score
