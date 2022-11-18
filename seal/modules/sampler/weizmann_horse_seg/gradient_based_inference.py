from typing import (
    List,
    Tuple,
    Union,
    Dict,
    Any,
    Optional,
    Callable,
    Generator,
)
from seal.modules.sampler import Sampler, GradientBasedInferenceSampler
import torch


@Sampler.register(
    "gradient-based-inference-weizmann-horse-seg-36crops", constructor="from_partial_objects"
)
class GradientBasedInferenceWeizmannHorseSeg36CropsSampler(GradientBasedInferenceSampler):

    def forward(
        self,
        x: Any,
        labels: Optional[
            torch.Tensor
        ],  #: If given will have shape (batch, ...)
        buffer: Dict,
        init_samples: torch.tensor = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # print("Forward in GBI 36crops.", flush=True)

        assert len(x.size()) == 5 and x.size()[1] == 36 # (b, 36, 3, 24, 24)
        x = x.view(-1, *x.size()[-3:]) # (b*36, 3, 24, 24)
        init = self.get_initial_output(x) # does not feed labels as reference

        # switch of gradients on parameters using context manager
        with self.no_param_grad():
            loss_fn = self.get_loss_fn(
                x,
                labels, # useless for weizmann horse seg eval
                buffer,
            )  #: Loss function will expect labels in form (batch, num_samples or 1, ...)
            (
                trajectory,  # new List[Tensor(batch, num_init_samples, ...)]
                loss_values_tensors,
                loss_values,
            ) = self.gradient_descent_loop(
                init,
                loss_fn,
                self.stopping_criteria,
                self.output_space.projection_function_,
            )
        loss_values_tensor = torch.tensor(loss_values)

        y_hat = self.get_samples_from_trajectory(
            trajectory, loss_values_tensors, loss_values
        ) # (batch, num_samples=1, ...)
        assert y_hat.size()[1] == 1

        return (
            y_hat.view(-1, 36, *y_hat.size()[2:]),
            None,
            torch.mean(loss_values_tensor),
        )