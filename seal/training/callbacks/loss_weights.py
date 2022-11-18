from typing import List, Tuple, Union, Dict, Any, Optional
import torch
from allennlp.common.lazy import Lazy
from allennlp.common.checks import ConfigurationError

from allennlp.data import TensorDict


from allennlp.training import (
    GradientDescentTrainer,
)
from allennlp.training.callbacks import TrainerCallback

import warnings
import logging

logger = logging.getLogger(__name__)


@TrainerCallback.register("lossweight-set-callback")
class TurnOnLossAfterEpochs(TrainerCallback):
    """
    This callback sets provided loss index (losses in the loss_idx_list)
    to be turned on/off if `model.epoch`> self.epoch_to_turn_on which can be read inside `forward()`.
    This callback lets you pass to the `GradientDescentTrainer` to access the current epoch number in your model during training.
    The losses in loss_idx_list will be initially set to 0 and turned on after trainig few epochs (self.epoch_to_turn_on).
    """

    def __init__(
        self,
        serialization_dir: str,
        loss_idx_list: Optional[List[int]] = None,
        epoch_to_turn_on: Optional[List[int]] = None,
        initial_weight_list: Optional[List[int]] = None,
    ) -> None:
        super().__init__(
            serialization_dir=serialization_dir,
        )
        self.loss_idx_list = loss_idx_list
        self.epoch_to_turn_on = epoch_to_turn_on

        if loss_idx_list is not None:
            if epoch_to_turn_on is not None:  # both provided.
                if len(loss_idx_list) != len(epoch_to_turn_on):
                    raise ConfigurationError(
                        "`epoch_to_turn_on` (List) should have the same length with `loss_idx_list`."
                    )
                else:
                    self.initial_weight_list = [0.0] * len(loss_idx_list)
            else:  # just loss_idx_list provided.
                raise ConfigurationError(
                    "`epoch_to_turn_on` (List) should be specified when `loss_idx_list` is specified."
                )
        elif epoch_to_turn_on is not None:  # just epoch_to_turn_on provided.
            raise ConfigurationError(
                "`loss_idx_list` (List) should be specified when `epoch_to_turn_on` is specified."
            )

    def get_loss_weights_then_set0(self, trainer: "GradientDescentTrainer"):
        for loss_idx in self.loss_idx_list:
            self.initial_weight_list[
                loss_idx
            ] = trainer.model.inference_module.loss_fn.loss_weights[loss_idx]
            trainer.model.inference_module.loss_fn.loss_weights[loss_idx] = 0

    def set_loss_weights(self, trainer: "GradientDescentTrainer", loss_idx):
        trainer.model.inference_module.loss_fn.loss_weights[
            loss_idx
        ] = self.initial_weight_list[loss_idx]

    def on_start(
        self,
        trainer: "GradientDescentTrainer",
        is_primary: bool = True,
        **kwargs,
    ) -> None:
        super().on_start(
            trainer, is_primary, **kwargs
        )  # --> trainer.model.epoch = 0  # type: ignore[assignment]
        self.get_loss_weights_then_set0(trainer)

    def on_epoch(
        self,
        trainer: "GradientDescentTrainer",
        metrics: Dict[str, Any],
        epoch: int,
        is_primary: bool = True,
        **kwargs,
    ) -> None:
        """
        Overriding on_epoch to control the weights.
        """
        super().on_epoch(
            trainer, metrics, epoch, is_primary, **kwargs
        )  
        if (
            self.loss_idx_list is not None
            and self.epoch_to_turn_on is not None
        ):
            for i, epoch_thresh in enumerate(self.epoch_to_turn_on):
                if trainer.model.epoch > epoch_thresh:
                   self.set_loss_weights(trainer, self.loss_idx_list[i]) #trainer.model.inference_module.loss_fn.loss_weights[loss_idx] = self.initial_weight_list[loss_idx]

@TrainerCallback.register("scoreloss-smooth-increase-callback")
class IncreaseScoreLossSmooth(TrainerCallback):
    """
    This callback sets provided loss index (losses in the loss_idx_list) 
    The losses in loss_idx_list will be initially set to 0 and turned on after trainig few epochs (self.epoch_to_turn_on).
    """
    def __init__(
        self,
        serialization_dir: str,
        score_loss_idx: int = 0,
        decay_rate: float = 0.95,
    ) -> None:
        super().__init__(
            serialization_dir=serialization_dir,
        )
        self.score_loss_idx = score_loss_idx
        self.score_rate = 1.0
        self.decay_rate = decay_rate
        self.initial_score_weight = 0.0

    def on_start(
        self, trainer: "GradientDescentTrainer", is_primary: bool = True, **kwargs
    ) -> None:
        """
        Gather provided weight for score loss and set it to 0 on beginnnig.
        """
        self.initial_score_weight = trainer.model.inference_module.loss_fn.loss_weights[self.score_loss_idx] 
        trainer.model.inference_module.loss_fn.loss_weights[self.score_loss_idx] = 0.0

    def on_batch(
        self,
        trainer: "GradientDescentTrainer",
        batch_inputs: List[List[TensorDict]],
        batch_outputs: List[Dict[str, Any]],
        batch_metrics: Dict[str, Any],
        epoch: int,
        batch_number: int,
        is_training: bool,
        is_primary: bool = True,
        batch_grad_norm: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        Overriding on_epoch to control the weights.
        """
        self.score_rate = self.score_rate*self.decay_rate
        trainer.model.inference_module.loss_fn.loss_weights[self.score_loss_idx] = (1-self.score_rate)*self.initial_score_weight

@TrainerCallback.register("decrease-xtropy-callback")
class DecreaseXtropyLoss(IncreaseScoreLossSmooth):
    """
    This callback changes weights for the provided loss index (losses in the loss_idx_list).
    The losses in loss_idx_list will be initially set to 1 and become provided min_weight as training steps progress.
    --> change this class & IncreaseScoreLossSmooth to inherit from Imitation Learning call back and reduce rewriting common parts.
    """
    def __init__(
        self,
        serialization_dir: str,
        score_loss_idx: int = 0,
        xtropy_loss_idx: int = 1,
        xtropy_min_weight: float = 0.3,
        decay_rate: float = 0.95,
    ) -> None:
        super().__init__(
            serialization_dir=serialization_dir,
            score_loss_idx=score_loss_idx,
            decay_rate=decay_rate,
        )
        # additional cross entropy (xtropy) related variables.
        self.xtropy_loss_idx = xtropy_loss_idx
        self.xtropy_min_weight = xtropy_min_weight
        self.initial_xtropy_weight= 0.0

    def on_start(
        self, trainer: "GradientDescentTrainer", is_primary: bool = True, **kwargs
    ) -> None:
        """
        Gather provided weight for score loss and set it to 0 on beginnnig.
        """
        # self.initial_score_weight = trainer.model.inference_module.loss_fn.loss_weights[self.score_loss_idx] 
        self.initial_xtropy_weight = trainer.model.inference_module.loss_fn.loss_weights[self.xtropy_loss_idx] 
        # trainer.model.inference_module.loss_fn.loss_weights[self.score_loss_idx] = 0.0
        trainer.model.inference_module.loss_fn.loss_weights[self.xtropy_loss_idx] = 1.0

    def on_batch(
        self,
        trainer: "GradientDescentTrainer",
        batch_inputs: List[List[TensorDict]],
        batch_outputs: List[Dict[str, Any]],
        batch_metrics: Dict[str, Any],
        epoch: int,
        batch_number: int,
        is_training: bool,
        is_primary: bool = True,
        batch_grad_norm: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        Overriding on_epoch to control the weights.
        """
        self.score_rate = self.score_rate*self.decay_rate
        # trainer.model.inference_module.loss_fn.loss_weights[self.score_loss_idx] = (1-self.score_rate)*self.initial_score_weight
        trainer.model.inference_module.loss_fn.loss_weights[self.xtropy_loss_idx] = (
                                                                            max(self.xtropy_min_weight, self.score_rate)
                                                                            * self.initial_xtropy_weight
                                                                        )


# @TrainerCallback.register("imitation-learning-callback")
# class ImitationLearningLoss(IncreaseScoreLossSmooth):
#     """
#     This callback sets provided loss index (losses in the loss_idx_list) 
#     The losses in loss_idx_list will be initially set to 0 and turned on after trainig few epochs (self.epoch_to_turn_on).
#     """
#     def __init__(
#         self,
#         serialization_dir: str,
#         score_loss_idx: int = 0,
#         xtropy_loss_idx: int = 1,
#         xtropy_min_weight: float = 0.3,
#         decay_rate: float = 0.95,
#     ) -> None:
#         super().__init__(
#             serialization_dir=serialization_dir,
#             score_loss_idx=score_loss_idx,
#             decay_rate=decay_rate,
#         )
#         # additional cross entropy (xtropy) related variables.
#         self.xtropy_loss_idx = xtropy_loss_idx
#         self.xtropy_min_weight = xtropy_min_weight
#         self.initial_xtropy_weight= 0.0

#     def on_start(
#         self, trainer: "GradientDescentTrainer", is_primary: bool = True, **kwargs
#     ) -> None:
#         """
#         Gather provided weight for score loss and set it to 0 on beginnnig.
#         """
#         self.initial_score_weight = trainer.model.inference_module.loss_fn.loss_weights[self.score_loss_idx] 
#         self.initial_xtropy_weight = trainer.model.inference_module.loss_fn.loss_weights[self.xtropy_loss_idx] 
#         trainer.model.inference_module.loss_fn.loss_weights[self.score_loss_idx] = 0.0
#         trainer.model.inference_module.loss_fn.loss_weights[self.xtropy_loss_idx] = 1.0

#     def on_batch(
#         self,
#         trainer: "GradientDescentTrainer",
#         batch_inputs: List[List[TensorDict]],
#         batch_outputs: List[Dict[str, Any]],
#         batch_metrics: Dict[str, Any],
#         epoch: int,
#         batch_number: int,
#         is_training: bool,
#         is_primary: bool = True,
#         batch_grad_norm: Optional[float] = None,
#         **kwargs: Any,
#     ) -> None:
#         """
#         Overriding on_epoch to control the weights.
#         """
#         self.score_rate = self.score_rate*self.decay_rate
#         trainer.model.inference_module.loss_fn.loss_weights[self.score_loss_idx] = (1-self.score_rate)*self.initial_score_weight
#         trainer.model.inference_module.loss_fn.loss_weights[self.xtropy_loss_idx] = (
#                                                                             max(self.xtropy_min_weight, self.score_rate)
#                                                                             * self.initial_xtropy_weight
#                                                                         )


# @TrainerCallback.register("nceloss-sample-weight")
# class ReduceNCESampleWeights(TrainerCallback):
#     """
#     This controls the "sample_weight" in interporlated loss (NCERankingInterpolatedLoss in nce_loss.py).
#     """
#     def __init__(
#         self,
#         serialization_dir: str,
#         min_sample_weight: float = 0.25,
#         decay_rate: float = 0.95,
#     ) -> None:
#         super().__init__(
#             serialization_dir=serialization_dir,
#         )
#         self.min_sample_weight = min_sample_weight
#         self.decay_rate = decay_rate

#     def on_start(
#         self, trainer: "GradientDescentTrainer", is_primary: bool = True, **kwargs
#     ) -> None:
#         super().on_start(trainer, is_primary,**kwargs) # --> trainer.model.epoch = 0  # type: ignore[assignment]
#         trainer.model.loss_fn.sample_weight = 1.0 

#     def on_batch(
#         self,
#         trainer: "GradientDescentTrainer",
#         batch_inputs: List[List[TensorDict]],
#         batch_outputs: List[Dict[str, Any]],
#         batch_metrics: Dict[str, Any],
#         epoch: int,
#         batch_number: int,
#         is_training: bool,
#         is_primary: bool = True,
#         batch_grad_norm: Optional[float] = None,
#         **kwargs: Any,
#     ) -> None:
#         # do everything as the parent does
#         trainer.model.loss_fn.sample_weight = max(
#                                             self.min_sample_weight, 
#                                             self.decay_rate*trainer.model.loss_fn.sample_weight
#                                             )