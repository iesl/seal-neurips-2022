from typing import List, Tuple, Union, Dict, Any, Optional
from allennlp.common.registrable import Registrable
import torch


class OutputSpace(Registrable):
    def get_samples(
        self,
        num_samples: Union[Tuple[int, ...], int],  #: includes the batch dim
        dtype: torch.dtype,
        device: torch.device,
        probabilities: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def get_mixed_samples(
        self,
        num_samples: Union[Tuple[int, ...], int],  #: does not include batch
        dtype: torch.dtype,
        reference: torch.Tensor,
        proportion_of_random_entries: float = 0.5,
        device: Optional[torch.device] = None,
        probabilities: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()  # type:ignore
    def projection_function_(self, inp: torch.Tensor) -> None:
        """Inplace projection function"""
        raise NotImplementedError
