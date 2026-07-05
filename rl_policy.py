"""Shared-trunk actor-critic network for guitar tab assignment."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from guitar_env import MAX_CANDIDATES, STATE_DIM


class ActorCritic(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(STATE_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(hidden, MAX_CANDIDATES)
        self.critic_head = nn.Linear(hidden, 1)

    def forward(
        self, state: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        state : (B, STATE_DIM)
        mask  : (B, MAX_CANDIDATES) — True where action is valid
        Returns log_probs (B, MAX_CANDIDATES) and value (B,).
        """
        x = self.trunk(state)
        logits = self.actor_head(x).masked_fill(~mask, float("-inf"))
        log_probs = F.log_softmax(logits, dim=-1)
        value = self.critic_head(x).squeeze(-1)
        return log_probs, value

    def act(
        self, state: np.ndarray, mask: np.ndarray,
        logit_bias: np.ndarray | None = None,
    ) -> tuple[int, float, float]:
        """Sample one action during rollout. Returns (action, log_prob, value).

        logit_bias: optional per-action bonus added to logits before softmax,
        used at inference time to nudge toward same-string on fast transitions.
        """
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            m = torch.from_numpy(np.asarray(mask, dtype=np.bool_)).unsqueeze(0)
            x = self.trunk(s)
            logits = self.actor_head(x).masked_fill(~m, float("-inf"))
            if logit_bias is not None:
                logits = logits + torch.tensor(logit_bias, dtype=torch.float32).unsqueeze(0)
            log_probs = F.log_softmax(logits, dim=-1)
            value = self.critic_head(x).squeeze(-1)
            probs = log_probs.exp()[0]
            if not probs.isfinite().any() or probs.sum() == 0:
                # Fallback: uniform over valid actions (should not happen after env filtering)
                probs = m[0].float()
            action = torch.multinomial(probs, 1).item()
        return action, log_probs[0, action].item(), value.item()
