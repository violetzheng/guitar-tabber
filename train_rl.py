"""PPO training for string+fret assignment.

Each episode is one pass through the full note sequence in notes.json.
Stochasticity comes from the policy sampling different (string, fret) choices;
the reward signal comes purely from playability rules (no ground-truth labels used).

For a dataset of songs, pass multiple --notes arguments and the env will be
re-sampled per episode (not yet implemented here — single-file training only).
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from guitar_env import GuitarTabEnv, MAX_CANDIDATES
from rl_policy import ActorCritic

# PPO hyperparameters
GAMMA = 0.99
LAM = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
LR = 3e-4
PPO_EPOCHS = 4
EPISODES_PER_BATCH = 20
NUM_ITERATIONS = 300


# ------------------------------------------------------------------
# Rollout collection
# ------------------------------------------------------------------

def collect_episodes(env: GuitarTabEnv, policy: ActorCritic, n: int):
    states, actions, log_probs_old = [], [], []
    rewards, values, masks, dones = [], [], [], []

    for _ in range(n):
        obs, valid_mask = env.reset()
        done = False
        while not done:
            action, lp, val = policy.act(obs, valid_mask)
            next_obs, reward, done, next_mask = env.step(action)

            states.append(obs)
            actions.append(action)
            log_probs_old.append(lp)
            rewards.append(reward)
            values.append(val)
            masks.append(valid_mask.copy())
            dones.append(done)

            obs, valid_mask = next_obs, next_mask

    return states, actions, log_probs_old, rewards, values, masks, dones


# ------------------------------------------------------------------
# Generalised Advantage Estimation
# ------------------------------------------------------------------

def compute_gae(rewards, values, dones):
    advantages = np.zeros(len(rewards), dtype=np.float32)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        if dones[t]:
            next_val = 0.0
            gae = 0.0   # don't bleed across episode boundaries
        else:
            next_val = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + GAMMA * next_val - values[t]
        gae = delta + GAMMA * LAM * gae
        advantages[t] = gae
    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


# ------------------------------------------------------------------
# PPO update
# ------------------------------------------------------------------

def ppo_update(policy, optimizer, states, actions, log_probs_old,
               advantages, returns, masks):
    states_t = torch.tensor(np.array(states), dtype=torch.float32)
    actions_t = torch.tensor(actions, dtype=torch.long)
    lp_old_t = torch.tensor(log_probs_old, dtype=torch.float32)
    adv_t = torch.tensor(advantages, dtype=torch.float32)
    ret_t = torch.tensor(returns, dtype=torch.float32)
    masks_t = torch.from_numpy(np.array(masks, dtype=np.bool_))

    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    actor_loss_total = value_loss_total = 0.0
    for _ in range(PPO_EPOCHS):
        log_probs, values = policy(states_t, masks_t)
        new_lp = log_probs.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        ratio = (new_lp - lp_old_t).exp()
        surr = torch.min(
            ratio * adv_t,
            ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * adv_t,
        )
        actor_loss = -surr.mean()
        value_loss = F.mse_loss(values, ret_t)

        probs = log_probs.exp()
        # Use log_probs directly (not probs.log()): avoids 1/0 gradient at masked
        # positions where probs=0 and log(0)=-inf would produce NaN in backward.
        entropy = -(probs * log_probs.masked_fill(~masks_t, 0.0)).sum(-1).mean()

        loss = actor_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()

        actor_loss_total += actor_loss.item()
        value_loss_total += value_loss.item()

    return actor_loss_total / PPO_EPOCHS, value_loss_total / PPO_EPOCHS


# ------------------------------------------------------------------
# Main training loop
# ------------------------------------------------------------------

def train(notes_json: str, save_path: str = "policy.pt",
          num_iterations: int = NUM_ITERATIONS) -> ActorCritic:
    env = GuitarTabEnv.from_json(notes_json)
    if not env.notes:
        raise ValueError(f"No usable notes found in {notes_json}")
    print(f"Training on {len(env.notes)} notes from {notes_json}")

    policy = ActorCritic()
    optimizer = Adam(policy.parameters(), lr=LR)

    for it in range(1, num_iterations + 1):
        states, actions, lp_old, rewards, values, masks, dones = (
            collect_episodes(env, policy, EPISODES_PER_BATCH)
        )
        advantages, returns = compute_gae(rewards, values, dones)
        a_loss, v_loss = ppo_update(
            policy, optimizer, states, actions, lp_old,
            advantages, returns, masks,
        )

        if it % 50 == 0:
            mean_reward = sum(rewards) / EPISODES_PER_BATCH
            print(
                f"iter {it:4d} | mean ep reward {mean_reward:.3f}"
                f" | actor {a_loss:.4f} | value {v_loss:.4f}"
            )

    torch.save(policy.state_dict(), save_path)
    print(f"Saved policy → {save_path}")
    return policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notes_json", help="Path to notes.json from note_events.py")
    parser.add_argument("-o", "--output", default="policy.pt", help="Policy save path")
    parser.add_argument("--iterations", type=int, default=NUM_ITERATIONS)
    args = parser.parse_args()
    train(args.notes_json, args.output, args.iterations)
