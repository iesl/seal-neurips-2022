from typing import List, Tuple, Union, Dict, Any, Optional, overload
from seal.modules.sampler import (
    Sampler,
    SamplerModifier,
    InferenceNetSampler,
)
import torch
from seal.modules.score_nn import ScoreNN
from seal.modules.oracle_value_function import (
    OracleValueFunction,
)
from seal.modules.multilabel_classification_task_nn import (
    MultilabelTaskNN,
)


@Sampler.register("multi-label-inference-net-normalized")
@InferenceNetSampler.register(
    "multi-label-inference-net-normalized",
)
class MultiLabelNormalized(InferenceNetSampler):
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


@Sampler.register("multi-label-inference-net-normalized-or-sampled")
@InferenceNetSampler.register(
    "multi-label-inference-net-normalized-or-sampled"
)
class MultiLabelNormalizedOrSampled(InferenceNetSampler):
    """
    Samples during training and normalizes during evaluation.
    """

    def __init__(
        self, num_samples: int = 1, keep_probs: bool = True, **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.keep_probs = keep_probs
        self.num_samples = num_samples if not keep_probs else num_samples - 1

    @overload
    def normalize(self, y: None) -> None:
        ...

    @overload
    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        ...

    def normalize(self, y: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if y is not None:
            if self._mode == "sample":
                return self.generate_samples(y)
            else:  # inference
                return torch.sigmoid(y)
        else:
            return None

    def generate_samples(self, y: torch.Tensor) -> torch.Tensor:
        assert (
            y.dim() == 3
        ), "Output of inference_net should be of shape (batch, 1, ...)"
        assert (
            y.shape[1] == 1
        ), "Output of inference_net should be of shape (batch, 1, ...)"
        p = torch.sigmoid(y).squeeze(1)  # (batch, num_labels)
        samples = torch.transpose(
            torch.distributions.Bernoulli(probs=p).sample(  # type: ignore
                [self.num_samples]  # (num_samples, batch, num_labels)
            ),
            0,
            1,
        )  # (batch, num_samples, num_labels)

        if self.keep_probs:
            samples = torch.cat(
                (samples, p.unsqueeze(1)), dim=1
            )  # (batch, num_samples+1, num_labels)

        return samples

    @property
    def different_training_and_eval(self) -> bool:
        return True

    @property
    def is_normalized(self) -> bool:
        return True


@Sampler.register("multi-label-inference-net-normalized-or-continuous-sampled")
@InferenceNetSampler.register(
    "multi-label-inference-net-normalized-or-continuous-sampled"
)
class MultiLabelNormalizedOrContinuousSampled(MultiLabelNormalizedOrSampled):
    """
    Samples during training and normalizes during evaluation.

    The samples are themselves probability distributions instead of hard samples. We
    do this by adding gaussian noise in the logit space (before taking sigmoid).
    """

    def __init__(self, std: float = 1.0, **kwargs: Any):
        super().__init__(**kwargs)
        self.std = std

    def generate_samples(self, y: torch.Tensor) -> torch.Tensor:
        assert (
            y.dim() == 3
        ), "Output of inference_net should be of shape (batch, 1, ...)"
        assert (
            y.shape[1] == 1
        ), "Output of inference_net should be of shape (batch, 1, ...)"
        # add gaussian noise
        # y.shape == (batch, 1, num_labels)
        samples = torch.sigmoid(
            torch.normal(
                y.expand(
                    -1, self.num_samples, -1
                ),  # (batch, num_samples, num_labels)
                std=self.std,
            )
        )  # (batch, num_samples, num_labels)

        if self.keep_probs:
            samples = torch.cat(
                (samples, torch.sigmoid(y)), dim=1
            )  # (batch, num_samples+1, num_labels)

        return samples
