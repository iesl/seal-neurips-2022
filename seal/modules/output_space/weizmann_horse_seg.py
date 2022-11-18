from typing import List, Tuple, Union, Dict, Any, Optional
from allennlp.common.registrable import Registrable
from seal.modules.output_space import OutputSpace
import torch


@OutputSpace.register("weizmann-horse-seg")
class WeizmannHorseSegOutputSpace(OutputSpace):
    def __init__(
            self,
            default_value: Optional[float] = None,
            mix_method: Optional[str] = "datum", # added for weizmann horse, datum or scalar
    ):
        self.default_value = default_value
        self.mix_method = mix_method

    @torch.no_grad()  # type:ignore
    def projection_function_(self, inp: torch.Tensor) -> None:
        """Inplace projection function"""
        inp.clamp_(0, 1)

    def get_samples(
        self,
        num_samples: Union[Tuple[int, ...], int],  #: includes the batch dim
        dtype: torch.dtype,
        device: torch.device,
        probabilities: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if isinstance(num_samples, int):
            sample_size = (num_samples,) + (1, 24, 24,)
        else:
            sample_size = num_samples + (1, 24, 24,)

        if self.default_value is None:  # random
            return torch.rand(size=sample_size, dtype=dtype, device=device)
        else:
            return torch.ones(size=sample_size, dtype=dtype, device=device) * self.default_value

    def get_mixed_samples(
            self,
            num_samples: int,
            dtype: torch.dtype,
            reference: torch.Tensor,
            proportion_of_random_entries: float = 0.5,
            device: Optional[torch.device] = None,
            probabilities: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        num_samples = (reference.shape[0], num_samples)
        samples = self.get_samples(
            num_samples,
            dtype=dtype,
            device=device or reference.device,
        )

        reference = reference.unsqueeze(1).expand_as(samples).to(dtype=dtype)
        with torch.no_grad():  # type: ignore
            if self.mix_method == "datum": # this only makes sense for 1 sample per datum
                indices = torch.rand(reference.shape[0]).float() > proportion_of_random_entries
                samples[indices] = reference[indices]
            elif self.mix_method == "scalar":
                take_reference_mask = (torch.rand_like(samples) > proportion_of_random_entries)
                samples = samples.masked_scatter(take_reference_mask, reference[take_reference_mask])
            else:
                raise NotImplementedError

        return samples
