"""Implements mean average precision using sklearn"""
from typing import List, Tuple, Union, Dict, Any, Optional

from sklearn.metrics import average_precision_score
from allennlp.training.metrics import Metric, Average
import torch


@Metric.register("multilabel-classification-mean-avg-precision")
class MultilabelClassificationMeanAvgPrecision(Average):

    """Docstring for MeanAvgPrecision. """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, predictions: torch.Tensor, gold_labels: torch.Tensor) -> None:  # type: ignore
        labels, scores = [
            t.cpu().numpy()
            for t in self.detach_tensors(gold_labels, predictions)
        ]

        for single_example_labels, single_example_scores in zip(
            labels, scores
        ):
            avg_precision = average_precision_score(
                single_example_labels, single_example_scores
            )
            super().__call__(avg_precision)
