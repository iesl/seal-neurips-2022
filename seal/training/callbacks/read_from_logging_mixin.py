from typing import List, Tuple, Union, Dict, Any, Optional
from allennlp.training import GradientDescentTrainer
from allennlp.training.callbacks import TrainerCallback
from allennlp.data import TensorDict
from seal.modules.logging import (
    LoggedScalarScalar,
    LoggingMixin,
)


@TrainerCallback.register("logging-mixin-to-metrics")
class ReadFromLoggingMixin(TrainerCallback):
    """
    Reads scalar values from the model logs (LoggingMixin) and adds the readings to metrics.
    """

    def on_start(
        self,
        trainer: "GradientDescentTrainer",
        is_primary: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        This callback hook is called before the training is started.
        """
        self.trainer = trainer
        assert isinstance(
            trainer.model, LoggingMixin
        ), "Model should inherit from LoggingMixin to use this callback"

    def on_batch(
        self,
        trainer: "GradientDescentTrainer",
        batch_inputs: List[TensorDict],
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
        This callback hook is called after the end of each batch.
        """

        if is_primary:
            vals: Dict[str, Union[float, int]] = trainer.model.get_all(  # type: ignore
                reset=False, type_=(LoggedScalarScalar,)
            )
        batch_metrics.update(vals)

    def on_epoch(
        self,
        trainer: "GradientDescentTrainer",
        metrics: Dict[str, Any],
        epoch: int,
        is_primary: bool = True,
        **kwargs,
    ) -> None:
        """
        This callback hook is called after the end of each epoch.
        """
        pass
