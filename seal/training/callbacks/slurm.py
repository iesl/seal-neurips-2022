from typing import List, Tuple, Union, Dict, Any, Optional
from allennlp.training.callbacks import TrainerCallback
from allennlp.training.gradient_descent_trainer import GradientDescentTrainer
import os
import logging

logger = logging.getLogger(__name__)


@TrainerCallback.register("slurm")
class Slurm(TrainerCallback):
    def on_start(
        self,
        trainer: "GradientDescentTrainer",
        is_primary: bool = True,
        **kwargs: Any,
    ) -> None:
        self.trainer = trainer
        logger.info(
            "%s     |   %s  ",
            "Slurm Variable".ljust(50),
            "Value".ljust(50),
        )
        logger.info(
            "%s     |   %s  ",
            "------".center(50, "-"),
            "------".center(50, "-"),
        )

        for name, value in os.environ.items():
            if name.startswith("SLURM_"):
                logger.info(
                    "%s     |   %s  ",
                    name.replace("SLURM_", "", 1).ljust(50),
                    value.ljust(50),
                )
