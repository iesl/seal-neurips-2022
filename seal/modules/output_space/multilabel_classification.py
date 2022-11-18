from typing import List, Tuple, Union, Dict, Any, Optional
from allennlp.common.registrable import Registrable
from seal.modules.output_space import OutputSpace
import torch


@OutputSpace.register("multi-label-discrete")
class MultilabelDiscreteOutputSpace(OutputSpace):
    def __init__(self, num_labels: int, default_value: Optional[float] = None):
        self.num_labels = num_labels
        self.default_value = default_value

    @torch.no_grad()  # type:ignore
    def projection_function_(self, inp: torch.Tensor) -> None:
        """Inplace projection function"""
        inp.clamp_(0, 1)

    def get_samples(
        self,
        num_samples: Union[Tuple[int, ...], int],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        probabilities: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            probabilities: Tensor of shape ()
        Note:
            If `probabilities` is given, we use that instead of the `default_value`.
        Returns:
            samples of shape (*num_samples, num_labels)
        """

        if isinstance(num_samples, int):
            num_samples = (num_samples,) + (self.num_labels,)
        else:
            num_samples = num_samples + (self.num_labels,)

        if self.default_value is None:  # random
            return torch.randint(
                low=0, high=2, size=num_samples, dtype=dtype, device=device
            )
        else:
            return (
                torch.ones(size=num_samples, dtype=dtype, device=device)
                * self.default_value
            )

    def get_mixed_samples(
        self,
        num_samples: Union[
            Tuple[int, ...], int
        ],  #: does not include the batch
        dtype: torch.dtype,
        reference: torch.Tensor,
        proportion_of_random_entries: float = 0.5,
        device: Optional[torch.device] = None,
        probabilities: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert reference.dim() == 2

        if isinstance(num_samples, int):
            num_samples = (reference.shape[0], num_samples)
        else:
            num_samples = (reference.shape[0],) + num_samples
        # We will add 1s in the samples dimension(s)
        ref_reshape_size = (
            (reference.shape[0],)
            + ((1,) * (len(num_samples) - 1))
            + (reference.shape[1],)
        )

        random_samples = self.get_samples(
            num_samples,
            dtype=dtype,
            device=device or reference.device,
        )
        take_reference_mask = (
            torch.rand_like(random_samples) > proportion_of_random_entries
        )
        with torch.no_grad():  # type: ignore
            samples = random_samples.masked_scatter(
                take_reference_mask,
                reference.reshape(ref_reshape_size)
                .expand_as(random_samples)
                .to(dtype=dtype)[take_reference_mask],
            )

        return samples


@OutputSpace.register("multi-label-relaxed")
class MultilabelRelaxedOutputSpace(MultilabelDiscreteOutputSpace):
    def __init__(self, num_labels: int, default_value: Optional[float] = None):
        super().__init__(num_labels, default_value)

    def get_samples(
        self,
        num_samples: Union[Tuple[int, ...], int],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        probabilities: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(num_samples, int):
            num_samples = (num_samples,) + (self.num_labels,)
        else:
            num_samples = num_samples + (self.num_labels,)

        if self.default_value is None:
            return torch.rand(size=num_samples, dtype=dtype, device=device)
        else:
            return (
                torch.ones(size=num_samples, dtype=dtype, device=device)
                * self.default_value
            )
