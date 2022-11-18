"""Implements micro average precision using sklearn"""
from typing import List, Tuple, Union, Dict, Any, Optional

from sklearn.metrics import average_precision_score
from allennlp.training.metrics import Metric, Average
import torch


@Metric.register("multilabel-classification-micro-avg-precision")
class MultilabelClassificationMicroAvgPrecision(Metric):

    """Docstring for MicroAvgPrecision."""

    def __init__(self) -> None:
        super().__init__()
        self.predicted = torch.Tensor([])
        self.gold = torch.Tensor([])

    def __call__(self, predictions: torch.Tensor, gold_labels: torch.Tensor) -> None:  # type: ignore
        # predictions, gold_labels: (batch_size, labels)
        predictions, gold_labels = [
            t.cpu() for t in self.detach_tensors(predictions, gold_labels)
        ]
        self.predicted = torch.cat([self.predicted, predictions], dim=0)
        self.gold = torch.cat([self.gold, gold_labels], dim=0)

    def get_metric(self, reset: bool) -> float:
        labels, scores = [
            t.cpu().numpy()
            for t in self.detach_tensors(self.gold, self.predicted)
        ]
        micro_precision_score = average_precision_score(
            labels, scores, average="micro"
        )

        if reset:
            self.reset()

        return micro_precision_score

    def reset(self) -> None:
        self.predicted = torch.Tensor([])
        self.gold = torch.Tensor([])
