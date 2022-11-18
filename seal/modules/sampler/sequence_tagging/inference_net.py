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
from seal.modules.sequence_tagging_task_nn import (
    SequenceTaggingTaskNN,
)


@Sampler.register("sequence-tagging-inference-net-normalized")
@InferenceNetSampler.register(
    "sequence-tagging-inference-net-normalized",
)
class SequenceTaggingNormalized(InferenceNetSampler):
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
            return torch.softmax(y,dim=-1)
        else:
            return None

    @property
    def different_training_and_eval(self) -> bool:
        return False


@Sampler.register("sequence-tagging-inference-net-normalized-or-sampled")
@InferenceNetSampler.register(
    "sequence-tagging-inference-net-normalized-or-sampled"
)
class SequenceTaggingNormalizedOrSampled(InferenceNetSampler):
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
                return torch.softmax(y,dim=-1)
        else:
            return None

    def generate_samples(self, y: torch.Tensor) -> torch.Tensor:
        '''
        This function provides discrete samples drawn from the probability y.
        Called by self.normalize() which is callend in the forward() of sampler.
        In the forward, when this function is called, y has (batch, 1, ...)
        '''
        assert (
            y.dim() == 4
        ), "Output of inference_net should be of shape  (batch, 1, seq_len, num_labels)"
        assert (
            y.shape[1] == 1
        ), "Output of inference_net should be of shape  (batch, 1, seq_len, num_labels)"
        p = torch.softmax(y,dim=-1).squeeze(1)   # (batch, seq_len, num_labels)
        samples = torch.transpose(
            torch.distributions.categorical.Categorical(probs=p).sample(  # type: ignore, <-- logits=y is also possible.
                [self.num_samples]  # (num_samples, batch, seq_len)
            ),
            0,
            1,
        )  # (# batch, num_samples, seq_len)
        samples = torch.nn.functional.one_hot(samples,y.shape[-1]) # (batch, num_samples, seq_len, num_labels)

        if self.keep_probs:
            samples = torch.cat(
                (samples, p.unsqueeze(1)), dim=1 #  p: (batch, 1, seq_len, num_labels)
            )  # (batch, num_samples+1, seq_len, num_labels)

        return samples

    @property
    def different_training_and_eval(self) -> bool:
        return True

    @property
    def is_normalized(self) -> bool:
        return True


@Sampler.register("sequence-tagging-inference-net-normalized-or-continuous-sampled")
@InferenceNetSampler.register(
    "sequence-tagging-inference-net-normalized-or-continuous-sampled"
)
class SequenceTaggingNormalizedOrContinuousSampled(SequenceTaggingNormalizedOrSampled):
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
            y.dim() == 4
        ), "Output of inference_net should be of shape  (batch, 1, seq_len, num_labels)"
        assert (
            y.shape[1] == 1
        ), "Output of inference_net should be of shape  (batch, 1, seq_len, num_labels)"
        # add gaussian noise
        # y.shape == (batch, seq_len, num_labels)
        samples = torch.softmax(
            torch.normal(
                y.expand( # (batch, 1, seq_len, num_labels)
                    -1, self.num_samples, -1, -1
                ),  # (batch, num_samples, seq_len, num_labels)
                std=self.std,
            ),
            dim=-1
        )  # (batch, num_samples, seq_len, num_labels)

        if self.keep_probs:
            samples = torch.cat(
                (samples, torch.softmax(y, dim=-1)), dim=1
            )  # (batch, num_samples+1, num_labels)

        return samples
