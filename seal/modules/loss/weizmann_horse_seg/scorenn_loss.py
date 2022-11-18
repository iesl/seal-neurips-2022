from typing import List, Tuple, Union, Dict, Any, Optional, Literal
from seal.modules.loss import Loss, DVNLoss, NCELoss, NCERankingLoss
import torch


@Loss.register("weizmann-horse-seg-dvn-bce")
class WeizmannHorseSegDVNCrossEntropyLoss(DVNLoss):

    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        if len(y.size()) == 4 and y.size()[-3] == 2: # (b, c=2, h, w)
            y = y[..., 1, :, :].unsqueeze(-3).unsqueeze(-3) # only the horse channel
        return y # (b, n, c=1, h, w)

    def compute_loss(
            self,
            predicted_score: torch.Tensor,  # logits of shape (batch, num_samples)
            oracle_value: Optional[torch.Tensor],  # (batch, num_samples)
    ) -> torch.Tensor:
        # both oracle values and predicted scores are higher the better
        # The oracle value will be something like f1 or 1-hamming_loss
        # which will take values in [0,1] with best value being 1.
        # Predicted score are logits, hence bce with logit will
        # internally map them to [0,1]

        return torch.nn.functional.binary_cross_entropy_with_logits(
            predicted_score,
            oracle_value,
            reduction=self.reduction,
        )


@Loss.register("weizmann-horse-seg-nce-ranking-with-discrete-sampling")
class WeizmannHorseSegNCERankingLoss(NCERankingLoss):
    def __init__(
            self,
            sign: Literal["-", "+"] = "-",
            use_scorenn: bool = True,
            use_distance: bool = True,
            **kwargs: Any
    ):
        super().__init__(use_scorenn, **kwargs)
        self.sign = sign
        self.mul = -1 if sign == "-" else 1
        self.bce = torch.nn.BCELoss(reduction="none")
        self.use_distance = use_distance
        # when self.use_scorenn=False, the sign should always be +,
        # as we want to have P_0/\sum(P_i) rather than (1/P_0) /\sum(1/P_i)
        assert (sign == "+" if not self.use_scorenn else True)

    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        if y.size()[-3] == 2: # (b, c=2, h, w)
            y = y[..., 1, :, :].unsqueeze(-3).unsqueeze(-3) # only the horse channel
        return y # (b, n=1, c=1, h, w)

    def distance(
            self,
            samples: torch.Tensor, # (b, 1+n, c=1, h, w)
            probs: torch.Tensor, # (b, 1+n, c=1, h, w)
    ) -> torch.Tensor:  # (batch, num_samples)
        """
        mul*BCE(inp=probs, target=samples). Here mul is 1 or -1. If mul = 1 the ranking loss will
        use adjusted_score of score - BCE. (mul=-1 corresponds to standard NCE)

        Note:
            Remember that BCE = -y ln(x) - (1-y) ln(1-x). Hence if samples are discrete, then BCE = -ln Pn(x).
            So in that case sign of + in this class will result in adjusted_score = score - (- ln Pn) = score + ln Pn.
        """
        if not self.use_distance: # if not using distance then skip the bce computation.
            return torch.zeros([samples.shape[0], samples.shape[1]], dtype=torch.long, device=probs.device) # (b, 1+n)

        samples = samples.view(*samples.size()[:-3], -1)
        probs = probs.view(*probs.size()[:-3], -1)
        loss = torch.sum(self.bce(probs, samples), dim=-1)
        # print("nce -lnPn", loss.max(), loss.min())

        return self.mul * loss  # (b, 1+n)

    def sample(
            self,
            probs: torch.Tensor, # (b, n=1, c=1, h, w)
    ) -> torch.Tensor: # (b, n, c=1, h, w)
        """
        Discrete sampling from the Bernoulli distribution.
        """
        p = probs.squeeze(-4)
        # print("p", p.max(), p.min())
        samples = torch.distributions.Bernoulli(probs=p).sample([self.num_samples]) # (n, b, c=1, h, w)

        return samples.transpose(0, 1) # (b, n, c=1, h, w)