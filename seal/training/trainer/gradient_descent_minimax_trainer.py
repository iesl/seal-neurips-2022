import datetime
import logging
import math
import os
import re
import time
import warnings
from typing import (
    Optional,
    Union,
    List,
    Dict,
    Tuple,
    Any,
    Type,
    Callable,
    cast,
    Generator,
    Literal,
    Iterator,
    cast,
)
import contextlib
import torch
from collections import defaultdict
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist

from allennlp.common.checks import ConfigurationError, check_for_gpu
from allennlp.common import util as common_util, Tqdm, Lazy
from allennlp.data.data_loaders.data_loader import DataLoader, TensorDict
from allennlp.models.model import Model
from allennlp.training.callbacks import ConsoleLoggerCallback
from allennlp.training.callbacks.confidence_checks import (
    ConfidenceChecksCallback,
)
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.learning_rate_schedulers.learning_rate_scheduler import (
    LearningRateScheduler,
)
from allennlp.training.metric_tracker import MetricTracker
from allennlp.training.momentum_schedulers.momentum_scheduler import (
    MomentumScheduler,
)
from allennlp.training.moving_average import MovingAverage
from allennlp.training.optimizers import (
    Optimizer,
    MultiOptimizer,
    ParameterGroupsType,
)
from allennlp.training.trainer import (
    Trainer,
    TrainerCheckpoint,
)
from allennlp.training.gradient_descent_trainer import (
    GradientDescentTrainer,
    DEFAULT_CALLBACKS,
)
from allennlp.training.callbacks import TrainerCallback
from allennlp.training import util as training_util
from seal.models.base import ScoreBasedLearningModel
from seal.common import ModelMode
from collections.abc import MutableMapping

logger = logging.getLogger(__name__)

MODE_LITERALS_TYPE = Literal[
    ModelMode.UPDATE_TASK_NN.value, ModelMode.UPDATE_SCORE_NN.value
]


@Optimizer.register("minimax")
class MiniMaxOptimizer(Optimizer, MutableMapping):
    """
    Holds multiple optimizers as dictionary with string keys.
    Each it behaves as `torch.optim.Optimizer` but all the methods take in
    an extra parameter `opt_key` (aka `model_mode`) to identify the optimizer to act upon.


    Although we inherit from `MultiOptimizer` we do not assign parameters like `MultiOptimizer`.
    `MultiOptimizer` assigns parameters to individual optimizers by looking for `optimizer_name` key
    in the parameter group values. We will instead assign parameters by querying the model for appropriate parameters.

    Note:
        We are skipping the complex parameter grouping logic for now. It can be implemented if needed.
    """

    def __init__(
        self,
        model_parameters: List[Tuple[str, torch.nn.Parameter]],
        optimizers: Dict[
            MODE_LITERALS_TYPE,
            Lazy[Optimizer],
        ],
        parameter_groups: ParameterGroupsType = None,
    ):
        # split the parameters and assign them to the correct optimizer
        # Note: If a parameter does not have the model_mode attribute set,
        # then that parameter will not be assigned to any optimizer

        if parameter_groups is not None:
            raise ConfigurationError("parameter_groups are not supported.")
        unassigned_params = []
        named_params_: Dict[
            MODE_LITERALS_TYPE,
            List[Tuple[str, torch.nn.Parameter]],
        ] = defaultdict(list)

        for n, p in model_parameters:
            if not ModelMode.hasattr_model_mode(p):
                unassigned_params.append(n)

                continue
            mode_name = ModelMode.getattr_model_mode(p).value

            if mode_name not in optimizers:
                unassigned_params.append(n)

                continue
            mode_name_: MODE_LITERALS_TYPE = cast(
                MODE_LITERALS_TYPE, mode_name
            )  # no runtime effect
            named_params_[mode_name_].append((n, p))

        logger.info("Optimizer assignements are as follows.")

        for mode_name, params in named_params_.items():
            logger.info(
                f"Following parameters have been assigned to the {mode_name} optimizer"
            )

            for n, p in params:
                logger.info(f"{n}")
        logger.info(
            "Following parameters have not been assigned to any optimizer, hence will not be updated"
        )

        for n in unassigned_params:
            logger.info(f"{n}")

        self.optimizers = {
            mode_name: lazy_optimizer.construct(
                model_parameters=named_params_[mode_name]
            )
            for mode_name, lazy_optimizer in optimizers.items()
        }

        super().__init__([v for k, v in model_parameters], {})

    def __getitem__(self, key: MODE_LITERALS_TYPE) -> torch.optim.Optimizer:
        return self.optimizers[key]

    def __setitem__(
        self, key: MODE_LITERALS_TYPE, value: torch.optim.Optimizer
    ) -> None:
        self.optimizers[key] = value

    def __delitem__(self, key: MODE_LITERALS_TYPE) -> None:
        del self.optimizers[key]

    def __iter__(self) -> Iterator[torch.optim.Optimizer]:
        return iter(self.optimizers)

    def __len__(self) -> int:
        return len(self.optimizers)

    def zero_grad(
        self, opt_key: Optional[str] = None, set_to_none: bool = False
    ) -> None:
        if opt_key is not None:
            self.optimizers[opt_key].zero_grad(set_to_none=set_to_none)
        else:
            for k, v in self.optimizers.items():
                v.zero_grad(set_to_none=set_to_none)

    def step(
        self,
        opt_key: Optional[str] = None,
        closure: Optional[Dict[str, Callable]] = None,
    ) -> None:
        if opt_key is not None:
            self.optimizers[opt_key].step(closure=closure)
        else:
            for k, v in self.optimizers.items():
                v.step(closure=closure[k] if closure is not None else None)

    def state_dict(self) -> Dict:
        """
        Creates an object `optimizer_state_dict`, which is a dictionary mapping an optimizer key to its
        `state_dict`. This dictionary is used as the value for 'optimizer' in the 'training_states' dictionary in
        the `gradient_descent` `Trainer`, e.g.
        ```
        "optimizer" : {
            "optimizer1": `optimizer1_state_dict`,
            "optimizer2": `optimizer2_state_dict`
        }.
        ```
        """
        optimizer_state_dict = {
            f"{optimizer_key}_optimizer": optimizer.state_dict()
            for optimizer_key, optimizer in self.optimizers.items()
        }

        return optimizer_state_dict

    def load_state_dict(self, training_state: Dict[str, Any]) -> None:
        """
        Loads each optimizer's `state_dict`.
        """

        for optimizer_key, optimizer in self.optimizers.items():
            optimizer.load_state_dict(
                training_state[f"{optimizer_key}_optimizer"]
            )


class ChecksForGradientDescentMiniMaxTrainer(TrainerCallback):
    def on_start(
        self,
        trainer: "GradientDescentTrainer",
        is_primary: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        This callback hook is called before the training is started.
        """

        if not isinstance(trainer, GradientDescentMinimaxTrainer):
            raise ConfigurationError(
                "Use this callback with GradientDescentMinimaxTrainer only"
            )

        if trainer._use_amp:
            raise ConfigurationError(
                "AMP is not supported for GradientDescentMinimaxTrainer"
            )
        # Make sure that there are not parameters left for the default optimizer
        # The constructor of MultiOptimizer will remove "default" optimizer
        # if it does not get any parameter. This is what we check.

        if not isinstance(trainer.optimizer, MultiOptimizer):
            raise ConfigurationError(
                "Optimizer for GradientDescentMinimaxTrainer should be of type MultiOptimizer"
            )

        if "default" in trainer.optimizer.optimizers:
            raise ConfigurationError(
                "In 'GradientDescentMinimaxTrainer' all the parameters should be assigned to either min or max optimizer"
                "But the following param groups are not assigned to any and are left for default optimizer"
                f"{trainer.optimizer.optimizers['default'].param_groups}"
            )

        if trainer._distributed:
            # DP: In order to make it work with DDP, we need to wrap all three methods _forward(), update() and compute_score()
            # in the main `forward()` call because DDP only syncs __call__(). Even after that, there might some lingering
            # issues. Hence, for now, we won't bother supporting it.
            raise ConfigurationError(
                "MiniMaxTrainer and ScoreBasedLearningModel does not support DDP."
            )

        if trainer._num_gradient_accumulation_steps != 1:
            raise ConfigurationError("Gradient accumulation is not supported.")
        super().on_start(trainer, is_primary=is_primary, **kwargs)


@Trainer.register(
    "gradient_descent_minimax", constructor="from_partial_objects"
)
class GradientDescentMinimaxTrainer(Trainer):
    """
    A trainer for doing two step minimax learning with gradient descent.


    This trainer is based on the `GradientDescentTrainer` from AllenNLP.
    The only parameter we modify is the optimizer. It is no longer an
    instance of `torch.optim.Optimizer` but a specialized instance of `MultiOptimizer`.

    """

    def __init__(
        self,
        model: ScoreBasedLearningModel,
        optimizer: MiniMaxOptimizer,
        data_loader: DataLoader,
        patience: Optional[int] = None,
        validation_metric: Union[str, List[str]] = "-loss",
        validation_data_loader: DataLoader = None,
        num_epochs: int = 20,
        serialization_dir: Optional[str] = None,
        checkpointer: Checkpointer = None,
        cuda_device: Optional[Union[int, torch.device]] = None,
        grad_norm: Optional[Dict[MODE_LITERALS_TYPE, Optional[float]]] = None,
        grad_clipping: Optional[float] = None,
        learning_rate_schedulers: Optional[
            Dict[
                MODE_LITERALS_TYPE,
                LearningRateScheduler,
            ]
        ] = None,
        momentum_schedulers: Optional[
            Dict[MODE_LITERALS_TYPE, MomentumScheduler]
        ] = None,
        moving_average: Optional[MovingAverage] = None,
        callbacks: List[TrainerCallback] = None,
        num_gradient_accumulation_steps: int = 1,
        enable_default_callbacks: bool = True,
        run_confidence_checks: bool = True,
        num_steps: Dict[MODE_LITERALS_TYPE, int] = None,
        inner_mode: MODE_LITERALS_TYPE = ModelMode.UPDATE_SCORE_NN.value,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            serialization_dir=serialization_dir,
            cuda_device=cuda_device,
            distributed=False,  # hardcode
            local_rank=0,  # hardcode
            world_size=1,  # hardcode
        )

        if "run_sanity_checks" in kwargs:
            warnings.warn(
                "'run_sanity_checks' is deprecated, please use 'run_confidence_checks' instead.",
                DeprecationWarning,
            )
            run_confidence_checks = kwargs["run_sanity_checks"]

        # I am not calling move_to_gpu here, because if the model is
        # not already on the GPU then the optimizer is going to be wrong.
        self.model = model

        self.data_loader = data_loader
        self.data_loader.set_target_device(self.cuda_device)
        self._validation_data_loader = validation_data_loader

        if self._validation_data_loader is not None:
            self._validation_data_loader.set_target_device(self.cuda_device)
        self.optimizer = optimizer

        if patience is None:  # no early stopping
            if validation_data_loader is not None:
                logger.warning(
                    "You provided a validation dataset but patience was set to None, "
                    "meaning that early stopping is disabled"
                )
        elif (not isinstance(patience, int)) or patience <= 0:
            raise ConfigurationError(
                '{} is an invalid value for "patience": it must be a positive integer '
                "or None (if you want to disable early stopping)".format(
                    patience
                )
            )

        # For tracking is_best_so_far and should_stop_early
        self._metric_tracker = MetricTracker(validation_metric, patience)

        self._num_epochs = num_epochs

        self._checkpointer: Optional[Checkpointer] = checkpointer

        if checkpointer is None and serialization_dir is not None:
            self._checkpointer = Checkpointer(serialization_dir)
        grad_norm = grad_norm or {}
        self._grad_norm: Dict[
            MODE_LITERALS_TYPE, Optional[float]
        ] = defaultdict(lambda: None, **grad_norm)
        self._grad_clipping = grad_clipping

        self._learning_rate_schedulers = learning_rate_schedulers
        self._momentum_schedulers = momentum_schedulers
        self._moving_average = moving_average

        self._callbacks = callbacks or []
        default_callbacks = (
            list(DEFAULT_CALLBACKS) if enable_default_callbacks else []
        )

        if run_confidence_checks:
            default_callbacks.append(ConfidenceChecksCallback)

        for callback_cls in default_callbacks:
            for callback in self._callbacks:
                if callback.__class__ == callback_cls:
                    break
            else:
                self._callbacks.append(callback_cls(self._serialization_dir))

        self._num_gradient_accumulation_steps = num_gradient_accumulation_steps

        self._pytorch_model = self.model

        # training state management
        self._epochs_completed: int = 0
        self._start_after_epochs_completed: int = 0
        self._batches_in_epoch_completed: int = 0
        self._start_after_batches_in_epoch_completed: int = 0
        self._best_model_filename: Optional[str] = None

        # This is a kind of training state, but it is not serialized with the trainer state, because we can
        # re-create it with `epochs_completed` and `batches_in_epoch_completed`.
        self._total_batches_completed: int = 0
        self._num_steps: Dict[ModelMode, int]

        if num_steps is None:
            self._num_steps = {
                ModelMode.UPDATE_TASK_NN: 1,
                ModelMode.UPDATE_SCORE_NN: 1,
            }
        else:
            self._num_steps = {ModelMode(k): v for k, v in num_steps.items()}
        self.inner_mode: ModelMode = ModelMode(inner_mode)
        self.exit_code = 0

    def num_steps(self, mode: ModelMode) -> int:
        return self._num_steps[mode]

    @contextlib.contextmanager
    def no_grad_for_other_mode(
        self, mode: Optional[ModelMode]
    ) -> Generator[None, None, None]:
        if mode is not None:
            # switch off the gradients for the param for the other mode but first cache their requires grad
            requires_grad_map_cache = {
                name: param.requires_grad
                for name, param in self.model.named_parameters()
            }
            try:
                # first switch off grad for other model and on for current mode if it is asked to be.

                for n, p in self.model.named_parameters():
                    if mode.is_parameter_model_mode(p):  # type: ignore
                        p.requires_grad_(p.requires_grad and True)
                    else:
                        p.requires_grad_(False)
                yield
            finally:
                # set the requires_grad back to what it was.

                for n, p in self.model.named_parameters():
                    p.requires_grad_(requires_grad_map_cache[n])
        else:  # if mode==None, no-op.
            try:
                yield
            finally:
                pass

    def rescale_gradients(self, mode: Optional[ModelMode] = None) -> float:
        """
        Performs gradient rescaling. Is a no-op if gradient rescaling is not enabled.

        Returns the norm of the gradients.
        """

        if mode is not None:

            def param_iter() -> Iterator[torch.nn.Parameter]:
                yield from self.model.parameters_for_model_mode(
                    mode  # type:ignore[arg-type]
                )

        else:

            def param_iter() -> Iterator[torch.nn.Parameter]:
                yield from self.model.parameters()

        parameters_to_clip = [p for p in param_iter() if p.grad is not None]
        _grad_norm: Optional[float] = self._grad_norm[
            mode.value if mode is not None else None
        ]

        if _grad_norm is not None:
            return clip_grad_norm_(parameters_to_clip, _grad_norm)
        else:
            return torch.norm(
                torch.stack(
                    [torch.norm(p.grad.detach()) for p in parameters_to_clip]
                )
            )

    def batch_outputs(
        self,
        batch: TensorDict,
        for_training: bool,
        mode: Optional[ModelMode] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Calls model's forward with right mode on the given batch and returns the output dictionary that the model
        returns. Currently, we ignore `get_regularization_penalty()`.
        """
        # DP: use self._pytorch_model here for DDP
        with self.no_grad_for_other_mode(mode):
            output_dict = self.model(**batch, mode=mode)
        output_dict["mode"] = mode
        if for_training:
            if "loss" not in output_dict:
                raise RuntimeError(
                    "The model you are trying to optimize does not contain a"
                    " 'loss' key in the output."
                )

        return output_dict

    def batch_forward_backward(
        self,
        batch: TensorDict,
        mode: ModelMode,
        batch_group_len: int,
        batch_group_outputs: List,
    ) -> float:
        batch_outputs = self.batch_outputs(batch, for_training=True, mode=mode)
        batch_group_outputs.append(batch_outputs)
        loss = batch_outputs["loss"]

        if torch.isnan(loss):
            raise ValueError("nan loss encountered")
        loss = loss / batch_group_len

        batch_loss = loss.item()  # already div by batch_group_len

        # Skipping on_backward() callbacks
        loss.backward()  # type: ignore

        return batch_loss

    def batch_group_step(
        self, batch_group: List[TensorDict], mode: ModelMode
    ) -> Tuple[float, List[Dict[str, Any]]]:
        batch_group_outputs: List[Dict[str, Any]] = []
        batch_group_loss = 0.0
        batch_group_len = len(batch_group)

        for batch in batch_group:
            batch_group_loss += self.batch_forward_backward(
                batch, mode, batch_group_len, batch_group_outputs
            )

        batch_grad_norm = self.rescale_gradients(mode)

        if self._learning_rate_schedulers:
            if mode.value in self._learning_rate_schedulers:
                self._learning_rate_schedulers[mode.value].step_batch(
                    self._total_batches_completed + 1
                )
        self.optimizer.step(opt_key=mode.value)

        return batch_group_loss, batch_group_outputs

    @classmethod
    def get_metrics(
        cls,
        model: Model,
        total_inner_loss: float,
        total_outer_loss: float,
        total_reg_loss: Optional[float],
        batch_inner_loss: Optional[float],
        batch_outer_loss: Optional[float],
        batch_reg_loss: Optional[float],
        num_batches: int,
        reset: bool = False,
        world_size: int = 1,
        cuda_device: Union[int, torch.device] = torch.device("cpu"),
    ) -> Dict[str, float]:
        """
        Gets the metrics but sets `"loss"` to
        the total loss divided by the `num_batches` so that
        the `"loss"` metric is "average loss per batch".
        Returns the `"batch_loss"` separately.
        """
        metrics = model.get_metrics(reset=reset)

        if batch_inner_loss is not None:
            metrics["batch_inner_loss"] = batch_inner_loss
        metrics["inner_loss"] = (
            float(total_inner_loss / num_batches) if num_batches > 0 else 0.0
        )
        metrics["loss"] = (
            float(total_outer_loss / num_batches) if num_batches > 0 else 0.0
        )

        if total_reg_loss is not None:
            if batch_reg_loss is not None:
                metrics["batch_reg_loss"] = batch_reg_loss
            metrics["reg_loss"] = (
                float(total_reg_loss / num_batches) if num_batches > 0 else 0.0
            )

        return metrics

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Trains one epoch and returns metrics.
        """

        logger.info("Epoch %d/%d", epoch, self._num_epochs - 1)
        cpu_memory_usage = []

        for worker, memory in common_util.peak_cpu_memory().items():
            cpu_memory_usage.append((worker, memory))
            logger.info(
                f"Worker {worker} memory usage: {common_util.format_size(memory)}"
            )
        gpu_memory_usage = []

        for gpu, memory in common_util.peak_gpu_memory().items():
            gpu_memory_usage.append((gpu, memory))
            logger.info(
                f"GPU {gpu} memory usage: {common_util.format_size(memory)}"
            )

        # regularization_penalty = self.model.get_regularization_penalty()
        regularization_penalty = None

        train_inner_loss = 0.0
        train_outer_loss = 0.0
        train_reg_loss = None if regularization_penalty is None else 0.0
        batch_reg_loss = None if regularization_penalty is None else 0.0

        # Set the model to "train" mode.
        self.model.train()

        # Get tqdm for the training batches
        batch_generator = iter(self.data_loader)
        batch_group_generator = common_util.lazy_groups_of(
            batch_generator, self._num_gradient_accumulation_steps
        )

        logger.info("Training")

        num_training_batches: Union[int, float]
        try:
            len_data_loader = len(self.data_loader)
            num_training_batches = math.ceil(
                len_data_loader / self._num_gradient_accumulation_steps
            )
        except TypeError:
            num_training_batches = float("inf")

        # Having multiple tqdm bars in case of distributed training will be a mess. Hence only the primary's
        # progress is shown

        if self._primary:
            batch_group_generator_tqdm = Tqdm.tqdm(
                batch_group_generator, total=num_training_batches
            )
        else:
            batch_group_generator_tqdm = batch_group_generator

        done_early = False

        for batch_group in batch_group_generator_tqdm:
            if done_early:
                break

            if (
                self._epochs_completed < self._start_after_epochs_completed
                or (
                    self._epochs_completed
                    == self._start_after_epochs_completed
                    and self._batches_in_epoch_completed
                    < self._start_after_batches_in_epoch_completed
                )
            ):
                self._batches_in_epoch_completed += 1
                self._total_batches_completed += 1

                continue
            # Extra/precautionary call to zero_grad
            # The required calls are in the loops.
            self.optimizer.zero_grad()

            batch_group_inner_loss = 0.0
            batch_group_outer_loss = 0.0
            batch_group_inner_outputs = []
            batch_group_outer_outputs = []

            num_inner_steps = self.num_steps(self.inner_mode)
            num_outer_steps = self.num_steps(self.inner_mode.flip())
            for outer_step in range(num_outer_steps):

                for inner_step in range(num_inner_steps):
                    # Check if optmizer for this mode is present

                    if self.inner_mode.value not in self.optimizer:
                        break
                    # we need to zero_grad before each optimization step.
                    self.optimizer.zero_grad(
                        opt_key=self.inner_mode.value, set_to_none=True
                    )
                    (
                        batch_group_inner_loss_,
                        batch_group_inner_outputs_,
                    ) = self.batch_group_step(
                        batch_group, mode=self.inner_mode
                    )
                    batch_group_inner_outputs += batch_group_inner_outputs_
                    batch_group_inner_loss += (
                        batch_group_inner_loss_ / num_inner_steps
                    )  # log avg inner loss
                    train_inner_loss += batch_group_inner_loss
                # outer step
                # Check if optmizer for this mode is present

                if self.inner_mode.flip().value not in self.optimizer:
                    continue

                self.optimizer.zero_grad(
                    opt_key=self.inner_mode.flip().value, set_to_none=True
                )
                (
                    batch_group_outer_loss_,
                    batch_group_outer_outputs_,
                ) = self.batch_group_step(
                    batch_group, mode=self.inner_mode.flip()
                )
                batch_group_outer_outputs += batch_group_outer_outputs_
                batch_group_outer_loss += (
                    batch_group_outer_loss_ / num_outer_steps
                )  # log avg outer loss
                train_outer_loss += batch_group_outer_loss

            # Update moving averages

            if self._moving_average is not None:
                self._moving_average.apply(self._total_batches_completed + 1)

            self._batches_in_epoch_completed += 1
            self._total_batches_completed += 1

            # Update the description with the latest metrics
            metrics = self.get_metrics(
                self.model,
                train_inner_loss,
                train_outer_loss,
                train_reg_loss,
                batch_group_inner_loss,
                batch_group_outer_loss,
                batch_reg_loss,
                self._batches_in_epoch_completed,
                world_size=self._world_size,
                cuda_device=self.cuda_device,
            )

            for callback in self._callbacks:
                callback.on_batch(
                    self,
                    batch_group,
                    batch_group_inner_outputs + batch_group_outer_outputs,
                    metrics,
                    epoch,
                    self._batches_in_epoch_completed,
                    is_training=True,
                    is_primary=self._primary,
                    batch_grad_norm=None,
                )

            if self._primary:
                # Updating tqdm only for the primary as the trainers wouldn't have one
                description = training_util.description_from_metrics(metrics)
                batch_group_generator_tqdm.set_description(
                    description, refresh=False
                )

                if self._checkpointer is not None:
                    self._checkpointer.maybe_save_checkpoint(
                        self,
                        self._epochs_completed,
                        self._batches_in_epoch_completed,
                    )

        if self._epochs_completed < self._start_after_epochs_completed or (
            self._epochs_completed == self._start_after_epochs_completed
            and self._batches_in_epoch_completed - 1
            < self._start_after_batches_in_epoch_completed
        ):
            metrics = {}
        else:
            metrics = self.get_metrics(
                self.model,
                train_inner_loss,
                train_outer_loss,
                train_reg_loss,
                batch_inner_loss=None,
                batch_outer_loss=None,
                batch_reg_loss=None,
                num_batches=self._batches_in_epoch_completed,
                reset=True,
                world_size=self._world_size,
                cuda_device=self.cuda_device,
            )

        for (worker, memory) in cpu_memory_usage:
            metrics["worker_" + str(worker) + "_memory_MB"] = memory / (
                1024 * 1024
            )

        for (gpu_num, memory) in gpu_memory_usage:
            metrics["gpu_" + str(gpu_num) + "_memory_MB"] = memory / (
                1024 * 1024
            )

        return metrics

    def _validation_loss(
        self, epoch: int
    ) -> Tuple[float, Optional[float], int]:
        """
        Computes the validation loss. Returns it and the number of batches.
        """
        logger.info("Validating")

        self._pytorch_model.eval()

        # Replace parameter values with the shadow values from the moving averages.

        if self._moving_average is not None:
            self._moving_average.assign_average_value()
        try:
            if self._validation_data_loader is not None:
                validation_data_loader = self._validation_data_loader
            else:
                raise ConfigurationError(
                    "Validation results cannot be calculated without a validation_data_loader"
                )

            regularization_penalty = None

            # Having multiple tqdm bars in case of distributed training will be a mess. Hence only the primary's
            # progress is shown

            if self._primary:
                val_generator_tqdm = Tqdm.tqdm(validation_data_loader)
            else:
                val_generator_tqdm = validation_data_loader

            batches_this_epoch = 0
            val_loss = 0.0
            val_batch_loss = 0.0
            val_reg_loss = None if regularization_penalty is None else 0.0
            val_batch_reg_loss = (
                None if regularization_penalty is None else 0.0
            )
            done_early = False

            for batch in val_generator_tqdm:

                batch_outputs = self.batch_outputs(batch, for_training=False)
                loss = batch_outputs.get("loss")
                reg_loss = batch_outputs.get("reg_loss")

                if loss is not None:
                    # You shouldn't necessarily have to compute a loss for validation, so we allow for
                    # `loss` to be None.  We need to be careful, though - `batches_this_epoch` is
                    # currently only used as the divisor for the loss function, so we can safely only
                    # count those batches for which we actually have a loss.  If this variable ever
                    # gets used for something else, we might need to change things around a bit.
                    batches_this_epoch += 1
                    val_batch_loss = loss.item()
                    val_loss += val_batch_loss

                    if reg_loss is not None:
                        val_batch_reg_loss = reg_loss.item()
                        val_reg_loss += val_batch_reg_loss  # type: ignore

                # Update the description with the latest metrics
                val_metrics = training_util.get_metrics(
                    self.model,
                    val_loss,
                    val_reg_loss,
                    val_batch_loss,
                    val_batch_reg_loss,
                    batches_this_epoch,
                    world_size=self._world_size,
                    cuda_device=self.cuda_device,
                )

                description = training_util.description_from_metrics(
                    val_metrics
                )

                if self._primary:
                    val_generator_tqdm.set_description(
                        description, refresh=False
                    )

                for callback in self._callbacks:
                    callback.on_batch(
                        self,
                        [batch],
                        [batch_outputs],
                        val_metrics,
                        epoch,
                        batches_this_epoch,
                        is_training=False,
                        is_primary=self._primary,
                    )

            return val_loss, val_reg_loss, batches_this_epoch
        finally:
            # Now restore the original parameter values.

            if self._moving_average is not None:
                self._moving_average.restore()

    def train(self) -> Dict[str, Any]:
        """
        Trains the supplied model with the supplied parameters.
        """
        try:
            self._restore_checkpoint()
        except RuntimeError as e:
            configuration_error = ConfigurationError(
                "Could not recover training from the checkpoint. Did you mean to output to "
                "a different serialization directory or delete the existing serialization "
                "directory?"
            )
            configuration_error.__cause__ = e
            raise configuration_error

        # Callbacks get their `on_start` call even when we're starting from a checkpoint.

        for callback in self._callbacks:
            callback.on_start(self, is_primary=self._primary)

        # Set default values in case of failure
        epoch = None
        metrics = None

        try:
            metrics, epoch = self._try_train()

            return metrics
        except:
            self.exit_code = 1
            raise # re-raise the exception
        finally:
            for callback in self._callbacks:
                callback.on_end(
                    self,
                    metrics=metrics,
                    epoch=epoch,
                    is_primary=self._primary,
                )

    def _try_train(self) -> Tuple[Dict[str, Any], int]:
        training_util.enable_gradient_clipping(self.model, self._grad_clipping)

        logger.info("Beginning training.")

        val_metrics: Dict[str, float] = {}
        metrics: Dict[str, Any] = {}
        training_start_time = None

        metrics["best_epoch"] = self._metric_tracker.best_epoch

        for key, value in self._metric_tracker.best_epoch_metrics.items():
            metrics["best_validation_" + key] = value

        for epoch in range(self._num_epochs):
            epoch_start_time = time.time()
            train_metrics = self._train_epoch(epoch)

            if self._epochs_completed < self._start_after_epochs_completed:
                # We're still catching up with the checkpoint, so we do nothing.
                # Note that we have to call _train_epoch() even when we know the epoch is skipped. We have to
                # read from the data loader, because the data loader and dataset readers might use randomness,
                # and we have to make sure we consume exactly the same instances in exactly the same way every
                # time we train, even when starting from a checkpoint, so that we update the randomness
                # generators in the same way each time.
                self._epochs_completed += 1
                self._batches_in_epoch_completed = 0

                continue

            if training_start_time is None:
                training_start_time = epoch_start_time

            # get peak of memory usage

            for key, value in train_metrics.items():
                if key.startswith("gpu_") and key.endswith("_memory_MB"):
                    metrics["peak_" + key] = max(
                        metrics.get("peak_" + key, 0), value
                    )
                elif key.startswith("worker_") and key.endswith("_memory_MB"):
                    metrics["peak_" + key] = max(
                        metrics.get("peak_" + key, 0), value
                    )

            this_epoch_val_metric: float = 0.0

            if self._validation_data_loader is not None:
                with torch.no_grad():
                    # We have a validation set, so compute all the metrics on it.
                    (
                        val_loss,
                        val_reg_loss,
                        num_batches,
                    ) = self._validation_loss(epoch)

                    # It is safe again to wait till the validation is done. This is
                    # important to get the metrics right.

                    val_metrics = training_util.get_metrics(
                        self.model,
                        val_loss,
                        val_reg_loss,
                        batch_loss=None,
                        batch_reg_loss=None,
                        num_batches=num_batches,
                        reset=True,
                        world_size=self._world_size,
                        cuda_device=self.cuda_device,
                    )

                    # Check validation metric for early stopping
                    this_epoch_val_metric = (
                        self._metric_tracker.combined_score(val_metrics)
                    )
                    self._metric_tracker.add_metrics(val_metrics)

            # Create overall metrics dict
            training_elapsed_time = time.time() - training_start_time
            metrics["training_duration"] = str(
                datetime.timedelta(seconds=training_elapsed_time)
            )
            metrics["epoch"] = epoch

            for key, value in train_metrics.items():
                metrics["training_" + key] = value

            for key, value in val_metrics.items():
                metrics["validation_" + key] = value

            if self._metric_tracker.is_best_so_far():
                # Update all the best_ metrics.
                # (Otherwise they just stay the same as they were.)
                metrics["best_epoch"] = epoch

                for key, value in val_metrics.items():
                    metrics["best_validation_" + key] = value

                self._metric_tracker.best_epoch_metrics = val_metrics

            if self._serialization_dir and self._primary:
                common_util.dump_metrics(
                    os.path.join(
                        self._serialization_dir, f"metrics_epoch_{epoch}.json"
                    ),
                    metrics,
                )

            # The Scheduler API is agnostic to whether your schedule requires a validation metric -
            # if it doesn't, the validation metric passed here is ignored.

            if self._learning_rate_schedulers:
                for _, sch in self._learning_rate_schedulers.items():
                    sch.step(this_epoch_val_metric)

            if self._momentum_schedulers:
                for _, msch in self._momentum_schedulers.items():
                    msch.step(this_epoch_val_metric)

            for callback in self._callbacks:
                callback.on_epoch(
                    self,
                    metrics=metrics,
                    epoch=epoch,
                    is_primary=self._primary,
                )

            self._epochs_completed += 1
            self._batches_in_epoch_completed = 0

            # The checkpointer saves state from the learning rate scheduler, momentum scheduler, moving
            # average, and callbacks, so we have to make sure those are updated before we save the
            # checkpoint here.

            if self._primary and self._checkpointer is not None:
                self._checkpointer.maybe_save_checkpoint(
                    self,
                    self._epochs_completed,
                    self._batches_in_epoch_completed,
                )
            # Wait for the primary process to finish saving the checkpoint

            if (
                self._primary
                and self._serialization_dir
                and self._metric_tracker.is_best_so_far()
            ):
                self._best_model_filename = os.path.join(
                    self._serialization_dir, "best.th"
                )

                if self._moving_average is None:
                    torch.save(
                        self.model.state_dict(), self._best_model_filename
                    )
                else:
                    self._moving_average.assign_average_value()
                    try:
                        torch.save(
                            self.model.state_dict(), self._best_model_filename
                        )
                    finally:
                        self._moving_average.restore()

            epoch_elapsed_time = time.time() - epoch_start_time
            logger.info(
                "Epoch duration: %s",
                datetime.timedelta(seconds=epoch_elapsed_time),
            )

            if self._metric_tracker.should_stop_early():
                logger.info("Ran out of patience. Stopping training.")

                break

            if epoch < self._num_epochs - 1:
                time_per_epoch = training_elapsed_time / (
                    (epoch + 1) - self._start_after_epochs_completed
                )
                # Note: If the first non-skipped epoch is half skipped (because it was checkpointed half-way
                # through), then this estimate is going to be optimistic.
                estimated_time_remaining = (
                    time_per_epoch * self._num_epochs
                ) - training_elapsed_time
                formatted_time = str(
                    datetime.timedelta(seconds=int(estimated_time_remaining))
                )
                logger.info(
                    "Estimated training time remaining: %s", formatted_time
                )
        else:
            epoch = self._num_epochs - 1

        # Load the best model state before returning

        if (
            self._best_model_filename is None
            or self._metric_tracker.is_best_so_far()
        ):
            self._finalize_model()
        else:
            # The model we're loading here has already been finalized.
            self.model.load_state_dict(torch.load(self._best_model_filename))

        return metrics, epoch

    def _finalize_model(self) -> None:
        """If we have a moving average, we have to finalize the model at the end of training."""

        if self._moving_average is not None:
            self._moving_average.assign_average_value()

    def get_checkpoint_state(self) -> TrainerCheckpoint:
        model_state = self.model.state_dict()

        # These are the training states we need to persist.
        training_states = {
            "version": 1,
            "metric_tracker": self._metric_tracker.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "callbacks": [cb.state_dict() for cb in self._callbacks],
            "epochs_completed": self._epochs_completed,
            "batches_in_epoch_completed": self._batches_in_epoch_completed,
            "best_model_filename": self._best_model_filename,
        }

        # If we have any of these optional objects, we should persist them too.

        if self._learning_rate_schedulers is not None:
            for name, sch in self._learning_rate_schedulers.items():
                training_states[
                    f"learning_rate_scheduler_{name}"
                ] = sch.state_dict()

        if self._momentum_schedulers is not None:
            for name, msch in self._momentum_schedulers.items():
                training_states[
                    "momentum_scheduler_{name}"
                ] = msch.state_dict()

        if self._moving_average is not None:
            training_states[
                "moving_average"
            ] = self._moving_average.state_dict()

        return TrainerCheckpoint(model_state, training_states)

    def _restore_checkpoint(self) -> None:
        """
        Restores the model and training state from the last saved checkpoint.
        This includes an epoch count and optimizer state, which is serialized separately
        from model parameters. This function should only be used to continue training -
        if you wish to load a model for inference/load parts of a model into a new
        computation graph, you should use the native Pytorch functions:
        `model.load_state_dict(torch.load("/path/to/model/weights.th"))`

        If `self._serialization_dir` does not exist or does not contain any checkpointed weights,
        this function will do nothing.
        """

        if self._checkpointer is None:
            return

        model_state, training_state = self._checkpointer.load_checkpoint()

        if len(model_state) <= 0 and len(training_state) <= 0:
            self._start_after_epochs_completed = 0
            self._start_after_batches_in_epoch_completed = 0
            self._best_model_filename = None

            return

        if training_state["version"] != 1:
            raise ValueError(
                f"This version of {self.__class__.__name__} only supports checkpoints of version 1. "
                f"Found version {training_state['version']}"
            )

        self.model.load_state_dict(model_state)
        self._metric_tracker.load_state_dict(training_state["metric_tracker"])
        self.optimizer.load_state_dict(training_state["optimizer"])

        for cb, state_dict in zip(
            self._callbacks, training_state["callbacks"]
        ):
            cb.load_state_dict(state_dict)

        if self._learning_rate_schedulers is not None:
            for name, sch in self._learning_rate_schedulers.items():
                sch.load_state_dict(
                    training_state[f"learning_rate_scheduler_{name}"]
                )

        if self._momentum_schedulers is not None:
            for name, msch in self._momentum_schedulers.items():
                msch.load_state_dict(
                    training_state[f"momentum_scheduler_{name}"]
                )

        if self._moving_average is not None:
            self._moving_average.load_state_dict(
                training_state["moving_average"]
            )

        self._start_after_epochs_completed = training_state["epochs_completed"]
        self._start_after_batches_in_epoch_completed = training_state[
            "batches_in_epoch_completed"
        ]
        self._best_model_filename = training_state["best_model_filename"]

    @classmethod
    def from_partial_objects(
        cls,
        model: ScoreBasedLearningModel,
        serialization_dir: str,
        data_loader: DataLoader,
        validation_data_loader: DataLoader = None,
        patience: int = None,
        validation_metric: Union[str, List[str]] = "-loss",
        num_epochs: int = 20,
        cuda_device: Optional[Union[int, torch.device]] = None,
        grad_norm: Optional[Dict[MODE_LITERALS_TYPE, Optional[float]]] = None,
        grad_clipping: float = None,
        num_gradient_accumulation_steps: int = 1,
        no_grad: List[str] = None,
        optimizer: Lazy[MiniMaxOptimizer] = Lazy(Optimizer.default),
        learning_rate_schedulers: Dict[
            MODE_LITERALS_TYPE, Lazy[LearningRateScheduler]
        ] = None,
        momentum_schedulers: Dict[
            MODE_LITERALS_TYPE, Lazy[MomentumScheduler]
        ] = None,
        moving_average: Lazy[MovingAverage] = None,
        checkpointer: Lazy[Checkpointer] = Lazy(Checkpointer),
        callbacks: List[Lazy[TrainerCallback]] = None,
        enable_default_callbacks: bool = True,
        run_confidence_checks: bool = True,
        num_steps: Dict[MODE_LITERALS_TYPE, int] = None,
        inner_mode: MODE_LITERALS_TYPE = ModelMode.UPDATE_SCORE_NN.value,
        **kwargs,
    ) -> Trainer:
        """
        This method exists so that we can have a documented method to construct this class using
        `FromParams`. If you are not using `FromParams` or config files, you can safely ignore this
        method.
        The reason we can't just use `__init__` with `FromParams` here is because there are
        sequential dependencies to this class's arguments.  Anything that has a `Lazy[]` type
        annotation needs something from one of the non-`Lazy` arguments.  The `Optimizer` needs to
        have the parameters from the `Model` before it's constructed, and the `Schedulers` need to
        have the `Optimizer`. Because of this, the typical way we construct things `FromParams`
        doesn't work, so we use `Lazy` to allow for constructing the objects sequentially.
        If you're not using `FromParams`, you can just construct these arguments in the right order
        yourself in your code and call the constructor directly.
        """

        if cuda_device is None:
            from torch import cuda

            if cuda.device_count() > 0:
                cuda_device = 0
            else:
                cuda_device = -1

        check_for_gpu(cuda_device)

        if cuda_device >= 0:
            # Moving model to GPU here so that the optimizer state gets constructed on
            # the right device.
            model = model.cuda(cuda_device)

        if no_grad:
            for name, parameter in model.named_parameters():
                if any(re.search(regex, name) for regex in no_grad):
                    parameter.requires_grad_(False)

        parameters = [
            [n, p] for n, p in model.named_parameters() if p.requires_grad
        ]
        optimizer_ = optimizer.construct(model_parameters=parameters)

        common_util.log_frozen_and_tunable_parameter_names(model)

        batches_per_epoch: Optional[int]
        try:
            batches_per_epoch = len(data_loader)
            batches_per_epoch = math.ceil(
                batches_per_epoch / num_gradient_accumulation_steps
            )
        except TypeError:
            batches_per_epoch = None

        moving_average_ = (
            None
            if moving_average is None
            else moving_average.construct(parameters=parameters)
        )
        learning_rate_schedulers_ = (
            None
            if learning_rate_schedulers is None
            else {
                name: sch.construct(
                    optimizer=optimizer_[name],
                    num_epochs=num_epochs,
                    num_steps_per_epoch=batches_per_epoch,
                )
                for name, sch in learning_rate_schedulers.items()
            }
        )
        momentum_schedulers_ = (
            None
            if momentum_schedulers is None
            else {
                name: msch.construct(
                    optimizer=optimizer_[name],
                    num_epochs=num_epochs,
                    num_steps_per_epoch=batches_per_epoch,
                )
                for name, msch in momentum_schedulers.items()
            }
        )
        checkpointer_ = checkpointer.construct(
            serialization_dir=serialization_dir
        )

        callbacks_: List[TrainerCallback] = []

        for callback_ in callbacks or []:
            callbacks_.append(
                callback_.construct(serialization_dir=serialization_dir)
            )

        return cls(
            model,
            optimizer_,
            data_loader,
            patience=patience,
            validation_metric=validation_metric,
            validation_data_loader=validation_data_loader,
            num_epochs=num_epochs,
            serialization_dir=serialization_dir,
            cuda_device=cuda_device,
            grad_norm=grad_norm,
            grad_clipping=grad_clipping,
            learning_rate_schedulers=learning_rate_schedulers_,
            momentum_schedulers=momentum_schedulers_,
            checkpointer=checkpointer_,
            moving_average=moving_average_,
            callbacks=callbacks_,
            num_gradient_accumulation_steps=num_gradient_accumulation_steps,
            enable_default_callbacks=enable_default_callbacks,
            run_confidence_checks=run_confidence_checks,
            num_steps=num_steps,
            inner_mode=inner_mode,
            **kwargs,
        )

    def get_best_weights_path(self) -> Optional[str]:
        return self._best_model_filename
