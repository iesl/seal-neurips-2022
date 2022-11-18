from allennlp.common import Registrable


class StoppingCriteria(Registrable):
    default_implementation = "number-of-steps"

    def __call__(self, step_number: int, loss_value: float) -> bool:
        raise NotImplementedError


@StoppingCriteria.register("number-of-steps")
class StopAfterNumberOfSteps(StoppingCriteria):
    def __init__(self, number_of_steps: int = 10):
        super().__init__()
        self.number_of_steps = number_of_steps

    def __call__(self, step_number: int, loss_value: float) -> bool:
        return step_number >= self.number_of_steps
