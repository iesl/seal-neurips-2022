"""Implements Average Rank using scipy.stats"""

import torch
from allennlp.training.metrics import Metric, Average
from scipy.stats import rankdata
import numpy as np


@Metric.register("multilabel-average-rank")
class MultilabelClassificationAvgRank(Average):

    """
    Computes average rank for the true label, given the energy scores for the samples and the true label
    Assumes, higher the score, better the rank.
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
        for single_example_labels, single_example_scores in zip(
            labels, scores
        ):
            scores_rank = rankdata(-single_example_scores, method='min')
            true_label_rank = np.sum(scores_rank * single_example_labels)
            super().__call__(true_label_rank)
