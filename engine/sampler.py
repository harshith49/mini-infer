"""Greedy and request-owned temperature/top-k/nucleus sampling."""
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    seed: int = 0

    def validate(self, vocab_size: int) -> None:
        if (not isinstance(self.temperature, (int, float))
                or not math.isfinite(self.temperature) or self.temperature < 0):
            raise ValueError('temperature must be finite and nonnegative')
        if type(self.top_k) is not int or not 0 <= self.top_k <= vocab_size:
            raise ValueError('top_k must be an integer in [0, vocab_size]')
        if (not isinstance(self.top_p, (int, float))
                or not math.isfinite(self.top_p) or not 0 < self.top_p <= 1):
            raise ValueError('top_p must be finite and in (0, 1]')
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError('seed must be an integer in [0, 2**63 - 1]')


def greedy(logits: torch.Tensor) -> torch.Tensor:
    """Return the highest-logit token ID; ties select the lowest ID."""
    return logits.argmax(dim=-1)


def sample(logits: torch.Tensor, params: SamplingParams,
           *, generator: torch.Generator) -> torch.Tensor:
    """Select one scalar token using the caller's generator, never a global seed."""
    if (logits.ndim != 1 or not logits.numel() or not logits.is_floating_point()
            or not torch.isfinite(logits).all()):
        raise ValueError('logits must be a finite nonempty rank-1 floating tensor')
    params.validate(logits.numel())
    if params.temperature == 0:
        return greedy(logits)
    # Center before dividing: tiny positive temperatures may map losers to -inf,
    # but the largest finite score stays zero and leaves a valid distribution.
    scores = logits.double()
    scores = (scores - scores.max()) / params.temperature
    if params.top_k:
        values, indices = scores.topk(params.top_k)
        scores = torch.full_like(scores, float('-inf')).scatter(0, indices, values)
    if params.top_p < 1:
        values, indices = scores.sort(descending=True)
        blocked = values.softmax(-1).cumsum(-1) > params.top_p
        # Keep the candidate crossing the threshold, including the largest one.
        blocked[1:] = blocked[:-1].clone()
        blocked[0] = False
        scores = torch.full_like(scores, float('-inf')).scatter(0, indices, values.masked_fill(blocked, float('-inf')))
    return torch.multinomial(scores.softmax(-1), 1, generator=generator).squeeze(0)
