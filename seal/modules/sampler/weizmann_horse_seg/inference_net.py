from typing import List, Tuple, Union, Dict, Any, Optional, overload
from seal.modules.sampler import (
    Sampler,
    SamplerModifier,
    InferenceNetSampler,
)
import torch
import torch.nn.functional as F
from seal.modules.loss import Loss
from seal.modules.score_nn import ScoreNN
from seal.modules.oracle_value_function import (
    OracleValueFunction,
)
from seal.modules.sequence_tagging_task_nn import (
    SequenceTaggingTaskNN,
)


@Sampler.register("weizmann-horse-seg-inference-net")
@InferenceNetSampler.register("weizmann-horse-seg-inference-net")
class WeizmannHorseSegInferenceNetSampler(InferenceNetSampler):

    def __init__(self, eval_loss_fn: Loss = None, **kwargs: Any):
        super(WeizmannHorseSegInferenceNetSampler, self).__init__(**kwargs)
        self.eval_loss_fn = eval_loss_fn

    @property
    def different_training_and_eval(self) -> bool:
        return True

    @property
    def is_normalized(self) -> bool:
        return True

    def normalize(self, y: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if y is None:
            return None

        if self.thirty_six_crops:
            y = y.view(-1, 36, *y.size()[-3:])

        if y.size()[-3] == 1: # (b, c=1, h, w)
            return torch.sigmoid(y)
        elif y.size()[-3] == 2: # (b, c=2, h, w)
            return torch.softmax(y, dim=-3) # keep channels, argmax during IoU metric calculation
        else:
            raise

    def forward(
        self,
        x: Any, # train (b, 3, 24, 24); test: None (b, 3, 32, 32) or thirty_six (b, 36, 3, 24, 24)
        labels: Optional[torch.Tensor],  # (b, c=1, h, w)
        buffer: Dict,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:

        self.thirty_six_crops = False
        if len(x.size()) == 5:
            assert x.size()[1] == 36 # (b, 36, 3, 24, 24)
            x = x.view(-1, *x.size()[-3:]) # (b*36, 3, 24, 24)
            self.thirty_six_crops = True

        y_hat = self.inference_nn(x) # (b, c=1 or 2, h, w) unnormalized logits
        # print("infnet inference module, max value in logits from tasknn", y_hat.max())

        if self.cost_augmented_layer is None or labels is None:
            y_cost_aug = None
        else:
            y_cost_aug = self.cost_augmented_layer(
                torch.cat((y_hat, labels.to(dtype=y_hat.dtype)), dim=-1), buffer,
            )

        if self.thirty_six_crops or x.size()[-1] == 32: # evaluation
            assert self.eval_loss_fn is not None
            loss = self.eval_loss_fn(x, labels, y_hat, y_cost_aug, buffer)
        else: # train
            loss = self.loss_fn(x, labels, y_hat, y_cost_aug, buffer)

        return (
                   self.normalize(y_hat), # (b, c=1 or 2, h, w) or (b, 36, c=1 or 2, 24, 24), no num_sample dim
                   self.normalize(y_cost_aug),
                   loss
        )

    @property
    def different_training_and_eval(self) -> bool:
        return False


@Sampler.register("weizmann-horse-seg-inference-net-normalized-or-continuous-sampled")
@InferenceNetSampler.register("weizmann-horse-seg-inference-net-normalized-or-continuous-sampled")
class WeizmannHorseSegInferenceNetNormalizedOrContinuousSampled(WeizmannHorseSegInferenceNetSampler):

    def __init__(self, num_samples: int = 1, keep_probs: bool = True, std: float = 1.0, **kwargs: Any):
        super().__init__(**kwargs)
        self.keep_probs = keep_probs
        self.num_samples = num_samples if not keep_probs else num_samples - 1
        self.std = std

    def normalize(self, y: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if y is None:
            return None

        if self._mode == "sample":
            assert not self.thirty_six_crops
            assert len(y.shape) == 4
            y = y.unsqueeze(-4)

            if y.size()[-3] == 1: # (b, n=1, c=1, h, w)
                samples = torch.sigmoid(
                    torch.normal(y.expand(-1, self.num_samples, -1, -1, -1), std=self.std)
                )
                if self.keep_probs:
                    samples = torch.cat((samples, torch.sigmoid(y)), dim=1)
                return samples

            elif y.size()[-3] == 2: # (b, n=1, c=2, h, w)
                samples = torch.softmax(
                    torch.normal(y.expand(-1, self.num_samples, -1, -1, -1), std=self.std),
                    dim=-3
                )
                if self.keep_probs:
                    samples = torch.cat((samples, torch.softmax(y, dim=-3)), dim=1)[:, :, 1, :, :].unsqueeze(-3)
                return samples # keep channels, argmax during IoU metric calculation

            else:
                raise

        else:
            if self.thirty_six_crops:
                y = y.view(-1, 36, *y.size()[-3:])

            if y.size()[-3] == 1: # (b, c=1, h, w)
                return torch.sigmoid(y)
            elif y.size()[-3] == 2: # (b, c=2, h, w)
                return torch.softmax(y, dim=-3) # keep channels, argmax during IoU metric calculation
            else:
                raise