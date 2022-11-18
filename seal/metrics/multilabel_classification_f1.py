"""Implements F1 using sklearn"""
from typing import List, Tuple, Union, Dict, Any, Optional

from sklearn.metrics import f1_score
from allennlp.training.metrics import Metric, Average
import torch
from seal.modules.oracle_value_function.multilabel_per_instance_f1 import (
    compute,
)


@Metric.register("multilabel-f1-score-with-threshold")
class MultilabelClassificationF1(Average):

    """Computes F1 score between true and predicted labels"""

    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__()
        self.threshold = threshold

    def __call__(
        self, predictions: torch.Tensor, gold_labels: torch.Tensor
    ) -> None:  # type: ignore

        labels, scores = [
            t.cpu().numpy()
            for t in self.detach_tensors(gold_labels, predictions)
        ]
        scores[scores < self.threshold] = 0
        scores[scores >= self.threshold] = 1

        for single_example_labels, single_example_scores in zip(
            labels, scores
        ):
            sample_f1 = f1_score(single_example_labels, single_example_scores)
            super().__call__(sample_f1)


@Metric.register("multilabel-relaxed-f1-score")
class MultilabelClassificationRelaxedF1(Average):

    """Computes F1 score between true and predicted labels.
    However, this metric uses the predictions between [0,1]
    as they are to produce f1 according to eq 7,9,10 in DVN
    paper.
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self, predictions: torch.Tensor, gold_labels: torch.Tensor
    ) -> None:  # type: ignore

        labels, scores = [
            t.cpu().numpy()
            for t in self.detach_tensors(gold_labels, predictions)
        ]
        per_instance_f1s = compute(
            gold_labels, predictions
        ).tolist()  # (batch,)

        for f1 in per_instance_f1s:
            super().__call__(f1)
