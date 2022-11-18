from typing import (
    List,
    Tuple,
    Union,
    Dict,
    Any,
    Optional,
    overload,
    Iterator,
    Literal,
    Generator,
)
from allennlp.common.registrable import Registrable
from allennlp.common.params import ConfigurationError
import torch
import contextlib
from allennlp.common.lazy import Lazy
from seal.common import ModelMode
from seal.modules.score_nn import ScoreNN
from seal.modules.oracle_value_function import (
    OracleValueFunction,
)
from seal.modules.task_nn import TaskNN
from seal.modules.loss import Loss
from seal.modules.logging import (
    LoggingMixin,
    LoggedValue,
    LoggedScalarScalar,
    LoggedScalarScalarSample,
    LoggedNPArrayNPArraySample,
)
import numpy as np


class Sampler(LoggingMixin, torch.nn.Module, Registrable):
    """
    Given input x, returns samples of shape `(batch, num_samples or 1,...)`
    and optionally their corresponding probabilities of shape `(batch, num_samples)`.
    **The sampler can do and return different things during training and test.**
    We want the probabilities specifically in the [[Minimum Risk Training for Neural Machine Translation|MRT setting]].

    The cases that sampler will cover include:
        1. Inference network or `TaskNN`, where we just take the input x and produce either a
            relaxed output of shape `(batch, 1, ...)` or samples of shape `(batch, num_samples, ...)`.
            Note, when we include `TaskNN` here, we also need to update its parameters, right here.
            So when sampler uses `TaskNN`, we also need to give it an instance of `Optimizer` to update its parameters.
        2. Cost-augmented inference module that uses `ScoreNN` and `OracleValueFunction` to produce a single relaxed output or samples.
        3. Adversarial sampler which again uses `ScoreNN` and `OracleValueFunction` to produce adversarial samples.
            (I see no difference between this and the cost augmented inference)
        4. Random samples biased towards `labels`.
        5. In the case of MRT style training, it can be beam search.
        6. In the case of vanilla feedforward model, one can just return the logits with shape `(batch, 1, ... )`
    """

    def parameters_with_model_mode(
        self, mode: ModelMode
    ) -> Iterator[torch.nn.Parameter]:
        yield from []

    def __init__(
        self,
        score_nn: Optional[ScoreNN] = None,
        oracle_value_function: Optional[OracleValueFunction] = None,
        mode: Literal["sample", "inference"] = "inference",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)  # type: ignore
        self.score_nn = score_nn
        self.oracle_value_function = oracle_value_function
        self._different_training_and_eval = False
        self._mode: Literal["sample", "inference"] = mode

    @contextlib.contextmanager
    def mode(
        self, mode: Literal["sample", "inference"]
    ) -> Generator[None, None, None]:
        current_mode = self._mode
        try:
            self._mode = mode
            yield
        finally:
            self._mode = current_mode

    @property
    def is_normalized(self) -> bool:
        """Whether the sampler produces normalized or unnormalized samples"""
        raise NotImplementedError

    @overload
    def normalize(self, y: None) -> None:
        ...

    @overload
    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        ...

    def normalize(self, y: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        raise NotImplementedError

    def forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor],
        buffer: Dict,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns:
            samples: Tensor of shape (batch, num_samples, ...)
            probabilities: None or tensor of shape (batch, num_samples)
            loss: None or tensor (scalar) loss.
        """
        raise NotImplementedError

    @property
    def different_training_and_eval(self) -> bool:
        return self._different_training_and_eval


class SamplerModifier(torch.nn.Module, Registrable):
    """Takes in samples and modifies them to produce
    samples again. Example use cases include:

        1. Sampling from a distribution.
        2. Normalizing unnormalized samples.
    """

    def forward(
        self, samples: torch.Tensor, samples_extra: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    @property
    def is_normalized(self) -> bool:
        raise NotImplementedError

    @property
    def different_training_and_eval(self) -> bool:
        raise NotImplementedError


class BasicSampler(Sampler):
    """
    Just as task_nn sampler.
    """

    def __init__(
        self,
        inference_nn: TaskNN,
        loss_fn: Loss,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.inference_nn = inference_nn
        self.loss_fn = loss_fn

    def parameters_with_model_mode(
        self, mode: ModelMode
    ) -> Iterator[torch.nn.Parameter]:
        yield from self.inference_nn.parameters()

    @overload
    def normalize(self, y: None) -> None:
        ...

    @overload
    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        ...

    def normalize(self, y: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        return y

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor],
        buffer: Dict,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:

        logits = self.inference_nn(x).unsqueeze(
            1
        )  # unormalized logits (batch, 1, ...)

        if labels is not None:
            # compute loss for logging.
            loss = self.loss_fn(
                x,
                labels.unsqueeze(1),  # (batch, num_samples or 1, ...)
                logits,
                logits,
                buffer,
            )
        else:
            loss = None

        return self.normalize(logits), self.normalize(logits), loss


class SamplerContainer(Sampler):
    """
    Abstract base class for sampler that uses constituent samplers.
    """

    def __init__(
        self,
        constituent_samplers: List[Sampler],
        score_nn: Optional[ScoreNN] = None,
        oracle_value_function: Optional[OracleValueFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(
            score_nn=score_nn,
            oracle_value_function=oracle_value_function,
            **kwargs,
        )
        self.constituent_samplers = torch.nn.ModuleList(constituent_samplers)

        if len(self.constituent_samplers) > 0:
            is_normalized = self.constituent_samplers[0].is_normalized
        else:
            is_normalized = None

        for s in self.constituent_samplers:
            self.logging_children.append(s)

        for s in self.constituent_samplers[1:]:
            assert (
                s.is_normalized == is_normalized
            ), f"is_normalized for {s} does not match {self.constituent_samplers[0]}"
        self._is_normalized = is_normalized

    @contextlib.contextmanager
    def mode(
        self, mode: Literal["sample", "inference"]
    ) -> Generator[None, None, None]:
        current_mode = self._mode
        constituent_samplers_current_modes = [
            s._mode for s in self.constituent_samplers
        ]
        try:
            self._mode = mode

            for s in self.constituent_samplers:
                s._mode = mode
            yield
        finally:
            self._mode = current_mode

            for s, m in zip(
                self.constituent_samplers, constituent_samplers_current_modes
            ):
                s._mode = m

    def append_sampler(self, sampler: Sampler) -> None:
        self.constituent_samplers.append(sampler)
        self.logging_children.append(sampler)

        if self._is_normalized is not None:
            assert (
                self._is_normalized == sampler.is_normalized
            ), f"is_normalized for the sampler being appended ({sampler}) is not same as that of other constituent samplers"
        else:
            self._is_normalized = sampler.is_normalized

    @property
    def is_normalized(self) -> bool:
        if self._is_normalized is not None:
            return self._is_normalized
        else:
            raise RuntimeError("Cannot determine the value.")


@SamplerContainer.register(
    "appending-container", constructor="from_partial_constituent_samplers"
)
@Sampler.register(
    "appending-container", constructor="from_partial_constituent_samplers"
)
class AppendingSamplerContainer(SamplerContainer):
    """
    Appends the samples generated by different samplers into one single set of samples.

    This class is useful, for example, when you want each batch to have samples
    from ground truth, gradient based inference as well as adversarial.
    """

    def __init__(
        self,
        constituent_samplers: List[Sampler],
        score_nn: Optional[ScoreNN] = None,
        oracle_value_function: Optional[OracleValueFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(
            constituent_samplers=constituent_samplers,
            score_nn=score_nn,
            oracle_value_function=oracle_value_function,
            **kwargs,
        )

    @classmethod
    def from_partial_constituent_samplers(
        cls,
        constituent_samplers: List[Lazy[Sampler]],
        score_nn: Optional[ScoreNN] = None,
        oracle_value_function: Optional[OracleValueFunction] = None,
        **kwargs: Any,
    ) -> Sampler:
        constructed_samplers = [
            sampler.construct(
                score_nn=score_nn, oracle_value_function=oracle_value_function
            )
            for sampler in constituent_samplers
        ]

        return cls(
            constructed_samplers,
            score_nn=score_nn,
            oracle_value_function=oracle_value_function,
            **kwargs,
        )

    def forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor],
        buffer: Dict,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        samples, probs, loss = list(
            zip(
                *[
                    sampler(x, labels, buffer, **kwargs)
                    for sampler in self.constituent_samplers
                ]
            )
        )  # samples: List[Tensor(batch, num_samples_for_sampler, ...)],
        # probs: List[Tensor(batch, num_samples_for_sampler, ...) or None]

        # DP: Currently we will not support combining probs
        # We can do it later if we need.
        all_samples = torch.cat(samples, dim=1)
        loss_ = torch.zeros_like(loss[0])

        for l_ in loss:
            if l_ is not None:
                loss_ = loss_ + l_

        return (all_samples, None, loss_)


@Sampler.register(
    "random-picking-container", constructor="from_partial_constituent_samplers"
)
class RandomPickingSamplerContainer(SamplerContainer):
    """
    On each call, picks one sampler randomly and returns samples from that sampler.

    This class is useful, for example, when you want to have multiple sampling strategies
    but only want samples from one of them in every batch.
    """

    def __init__(
        self,
        constituent_samplers: List[Sampler],
        probabilities: Optional[List[float]] = None,
        score_nn: Optional[ScoreNN] = None,
        oracle_value_function: Optional[OracleValueFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(
            constituent_samplers=constituent_samplers,
            score_nn=score_nn,
            oracle_value_function=oracle_value_function,
            **kwargs,
        )

        if probabilities is not None:
            assert len(probabilities) == len(self.constituent_samplers)
            # normalize
            total = sum(probabilities)
            self.probabilities = [p / total for p in probabilities]
        else:  # None
            total = len(self.constituent_samplers)
            self.probabilities = [1.0 / total] * total

    @classmethod
    def from_partial_constituent_samplers(
        cls,
        constituent_samplers: List[Lazy[Sampler]],
        probabilities: Optional[List[float]] = None,
        score_nn: Optional[ScoreNN] = None,
        oracle_value_function: Optional[OracleValueFunction] = None,
        **kwargs: Any,
    ) -> Sampler:
        constructed_samplers = [
            sampler.construct(
                score_nn=score_nn, oracle_value_function=oracle_value_function
            )
            for sampler in constituent_samplers
        ]

        return cls(
            constructed_samplers,
            probabilities=probabilities,
            score_nn=score_nn,
            oracle_value_function=oracle_value_function,
            **kwargs,
        )

    def forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor],
        buffer: Dict,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        sampler = np.random.choice(
            self.constituent_samplers, p=self.probabilities
        )

        return sampler(x, labels, buffer, **kwargs)


class SamplerWrapper(Sampler):
    """
    Just a wrapper that holds and calls a sampler.
    """

    def __init__(
        self,
        constituent_sampler: Sampler,
        score_nn: Optional[ScoreNN] = None,
        oracle_value_function: Optional[OracleValueFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(score_nn, oracle_value_function, **kwargs)
        self.constituent_sampler = constituent_sampler


@Sampler.register("with-modifier")
class SamplerWithModifier(Sampler):
    """
    Use this sampler to modify samples produced by another sampler.
    """

    def __init__(
        self,
        main_sampler: Sampler,
        sampler_modifier: SamplerModifier,
        score_nn: Optional[ScoreNN] = None,
        oracle_value_function: Optional[OracleValueFunction] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            score_nn=score_nn,
            oracle_value_function=oracle_value_function,
            **kwargs,
        )

        self.constituent_sampler = main_sampler
        self.sampler_modifier = sampler_modifier

    @property
    def different_training_and_eval(self) -> bool:
        return (
            self.constituent_sampler.different_training_and_eval
            or self.sampler_modifier.different_training_and_eval
        )

    @property
    def is_normalized(self) -> bool:
        if self.sampler_modifier:
            return self.sampler_modifier.is_normalized
        else:
            return self.constituent_sampler.is_normalized

    def forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor],
        buffer: Dict,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        y, y_extra = self.constituent_sampler(x, labels, buffer, **kwargs)

        if self.sampler_modifier:
            return self.sampler_modifier(y, y_extra), y_extra

        return y, y_extra


@Sampler.register("from-container")
class SamplerFromContainer(Sampler):
    """
    This class is designed to be used as an `inference_module`.
    It will pick one of the constituent samplers in the main `sampler`.
    """

    def __init__(
        self,
        main_sampler: SamplerContainer,
        index: int = -1,
        score_nn: Optional[ScoreNN] = None,
        oracle_value_function: Optional[OracleValueFunction] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            score_nn=score_nn,
            oracle_value_function=oracle_value_function,
            **kwargs,
        )

        assert isinstance(
            main_sampler, SamplerContainer
        ), "main_sampler has to be of type SamplerContainer"
        try:
            constituent_sampler = main_sampler.constituent_samplers[index]
        except IndexError as e:
            raise ConfigurationError(
                f"There is not constituent_sampler with index {index}"
            )
        self.constituent_sampler = constituent_sampler

    def forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor],
        buffer: Dict,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.constituent_sampler(x, labels, buffer, **kwargs)

    @property
    def is_normalized(self) -> bool:
        return self.constituent_sampler.is_normalized


@Sampler.register(
    "indexed-container", constructor="from_partial_constituent_samplers"
)
class IndexedSamplerContainer(SamplerContainer):
    """
    On each call, picks the sampler using the given index and returns samples from that sampler.
    """

    def __init__(
        self,
        constituent_samplers: List[Sampler],
        score_nn: Optional[ScoreNN] = None,
        oracle_value_function: Optional[OracleValueFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(
            constituent_samplers=constituent_samplers,
            score_nn=score_nn,
            oracle_value_function=oracle_value_function,
            **kwargs,
        )

    @classmethod
    def from_partial_constituent_samplers(
        cls,
        constituent_samplers: List[Lazy[Sampler]],
        score_nn: Optional[ScoreNN] = None,
        oracle_value_function: Optional[OracleValueFunction] = None,
        **kwargs: Any,
    ) -> Sampler:
        constructed_samplers = [
            sampler.construct(
                score_nn=score_nn, oracle_value_function=oracle_value_function
            )
            for sampler in constituent_samplers
        ]

        return cls(
            constructed_samplers,
            score_nn=score_nn,
            oracle_value_function=oracle_value_function,
            **kwargs,
        )

    def forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor],
        buffer: Dict,
        index: int = 0,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert (
            0 <= index < len(self.constituent_samplers)
        ), f"There is no constituent_sampler with index {index}"
        sampler = self.constituent_samplers[index]
        return sampler(x, labels, buffer, **kwargs)
