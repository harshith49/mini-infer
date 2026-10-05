"""Greedy sampling; stochastic strategies arrive with later serving milestones."""
import torch


def greedy(logits: torch.Tensor) -> torch.Tensor:
    """Return the highest-logit token ID; ties select the lowest ID."""
    return logits.argmax(dim=-1)
