from typing import List, Tuple, Union, Dict, Any, Optional, Literal
from seal.modules.loss import (
    Loss,
    NCELoss,
    NCERankingLoss,
)
import torch


def _normalize(y: torch.Tensor) -> torch.Tensor:
    return torch.softmax(y,dim=-1)


class SeqTagNCERankingLoss(NCERankingLoss):
    def __init__(self, 
                sign: Literal["-", "+"] = "-", 
                use_scorenn: bool = True,
                use_distance: bool = True,
                **kwargs: Any):
        super().__init__(use_scorenn, **kwargs)
        self.sign = sign
        self.mul = -1 if sign == "-" else 1
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="none") # works on logits, not prob.
        self.use_distance = use_distance
        # when self.use_scorenn=False, the sign should always be +,
        # as we want to have P_0/\sum(P_i) rather than (1/P_0) /\sum(1/P_i)
        assert (sign == "+" if not self.use_scorenn else True)  

    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        return _normalize(y)

    def distance(
        self,
        samples: torch.Tensor,  # (batch, num_samples, num_seq, num_labels)
        probs: torch.Tensor,  # (batch, num_samples, num_seq, num_labels) # expanded
    ) -> torch.Tensor:  # (batch, num_samples)
        """
        mul*CE(inp=probs, target=samples). Here mul is 1 or -1. If mul = 1 the ranking loss will
        use adjusted_score of score - CE. (mul=-1 corresponds to standard NCE)

        Note:
            Remember that CE = -y ln(x). Hence if samples are discrete, then CE = -ln Pn.
            So in that case sign of + in this class will result in adjusted_score = score - (- ln Pn) = score + ln Pn.
        """
        if not self.use_distance: # if not using distance then skip the CE computation.
            return torch.zeros([samples.shape[0], samples.shape[1]], dtype=torch.long, device=probs.device) # (batch,sample)
        
        def softCE(probability, target): # taskes care of both discrete/soft target.
            return -(target * torch.log(probability)).sum(dim=-1).sum(dim=-1)
            
        # return self.mul * torch.sum( torch.sum(
        #     self.cross_entropy(torch.log(probs), samples), #torch.log() to make the prob value to be logit.
        # dim=-1), dim=-1)  # (batch, num_samples) 

        return self.mul * softCE(probs, samples) 


@Loss.register("seqtag-nce-ranking-with-discrete-sampling")
class SeqTagNCERankingLossWithDiscreteSamples(SeqTagNCERankingLoss):
    def __init__(self, keep_probs = False, **kwargs: Any):
        super().__init__(**kwargs)
        self.keep_probs = keep_probs

    def sample(
        self,
        probs: torch.Tensor,  # (batch, 1, num_labels)
    ) -> torch.Tensor:  # (batch, num_samples, num_labels)
        """
        Discrete sampling from the softmax distribution.
        Very similar to SequenceTaggingNormalizedOrSampled.generate_samples() in inference_net.py
        except this function gets argumnet of probability as opposed to logit in the other.
        """
        assert (
            probs.dim() == 4
        ), "Output of inference_net should be of shape  (batch, 1, seq_len, num_labels)"
        assert (
            probs.shape[1] == 1
        ), "Output of inference_net should be of shape  (batch, 1, seq_len, num_labels)"

        p = probs.squeeze(1)   # (batch, seq_len, num_labels)
        samples = torch.transpose(
            torch.distributions.categorical.Categorical(probs=p).sample(  # type: ignore, <-- logits=y is also possible.
                [self.num_samples]  # (num_samples, batch, seq_len)
            ),
            0,
            1,
        )  # (# batch, num_samples, seq_len)
        samples = torch.nn.functional.one_hot(samples,probs.shape[-1]) # (batch, num_samples, seq_len, num_labels)

        if self.keep_probs:
            samples = torch.cat(
                (samples, probs), dim=1 #  p: (batch, 1, seq_len, num_labels)
            )  # (batch, num_samples+1, seq_len, num_labels)

        return samples


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    x should be in [0, 1]
    """

    return -torch.log((1.0 / (x + 1e-13)) - 1.0 + 1e-35)


@Loss.register("seqtag-nce-ranking-with-cont-sampling")
class SeqTagNCERankingLossWithContSamples(SeqTagNCERankingLoss):
    def __init__(self, std: float = 1.0, keep_probs = False, **kwargs: Any):
        super().__init__(**kwargs)
        self.std = std
        self.keep_probs = keep_probs

    def sample(
        self,
        probs: torch.Tensor,  # (batch, 1, num_labels)
    ) -> torch.Tensor:  # (batch, num_samples, num_labels)
        """
        Cont sampling from by adding gaussian noise to logits (acquired from torch.log()).
        Very similar to SequenceTaggingNormalizedOrContinuousSampled.generate_samples() in inference_net.py
        except this function gets argumnet of probability as opposed to logit in the other.
        """
        assert (
            probs.dim() == 4
        ), "Output of inference_net should be of shape  (batch, 1, seq_len, num_labels)"
        assert (
            probs.shape[1] == 1
        ), "Output of inference_net should be of shape  (batch, 1, seq_len, num_labels)"
        # add gaussian noise
        # y.shape == (batch, seq_len, num_labels)
        logits = torch.log(probs)
        samples = torch.softmax(
            torch.normal(
                logits.expand( # (batch, 1, seq_len, num_labels)
                    -1, self.num_samples, -1, -1
                ),  # (batch, num_samples, seq_len, num_labels)
                std=self.std,
            ),
            dim=-1
        )  # (batch, num_samples, seq_len, num_labels)

        if self.keep_probs:
            samples = torch.cat(
                (samples, probs), dim=1
            )  # (batch, num_samples+1, num_labels)

        return samples