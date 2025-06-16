"""PPO implementation for Sparrow Mahjong."""

import os
import time
from collections import defaultdict
from typing import List, Tuple, Dict

import jax
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm

from Training.env_management import (
    init_env,
    reset_random_seed,
    state_to_input,
    shuffled_to_normal,
    choose_next_move,
    tile_to_action,
    actions_to_hand,
)

# ============================================================================
# HYPERPARAMETERS - All configurable values in one place
# ============================================================================

# Environment & Model Architecture
INPUT_SIZE = 37
NUM_ACTIONS = 6
BATCH_SIZE = 200 
HIDDEN_SIZE = 32 
NUM_LSTM_LAYERS = 3
NUM_FC_LAYERS = 8
FC_HIDDEN_SIZE = 32
ENV_ID = "sparrow_mahjong"

# Training Configuration
MAX_UPDATES = 1750
LR = 1e-4 
GAMMA = 0.99
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PPO Specific
CLIP_EPSILON = 0.2
N_EPOCHS = 4  # number of PPO epochs per update
VALUE_CLIP = 0.2
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5

# Safety & Logging
MAX_EPISODE_STEPS = 100  # safety break for episodes
CHECKPOINT_FREQ = 100    # save checkpoint every N updates
VALID_ACTIONS_COUNT = 6  # expected number of valid actions per turn
ADVANTAGE_EPSILON = 1e-8 # small value for advantage normalization

# ============================================================================w

class ActorCriticLSTM(nn.Module):
    """Actor-Critic network with an LSTM body."""

    def __init__(
        self,
        input_size: int = 37,
        hidden_size: int = 128,
        num_lstm_layers: int = 1,
        num_fc_layers: int = 2,
        fc_hidden_size: int = 64,
        num_actions: int = 6,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
        )

        # Shared fully-connected stack
        fc_layers = []
        in_dim = hidden_size
        for _ in range(num_fc_layers):
            fc_layers.append(nn.Linear(in_dim, fc_hidden_size))
            fc_layers.append(nn.ReLU())
            in_dim = fc_hidden_size
        self.shared_mlp = nn.Sequential(*fc_layers)

        # Heads
        self.policy_head = nn.Linear(in_dim, num_actions)
        self.value_head = nn.Linear(in_dim, 1)

    def get_value(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Forward pass for critic only."""
        with torch.no_grad():
            lstm_out, _ = self.lstm(x, hidden)
            last_out = lstm_out[:, -1, :]
            z = self.shared_mlp(last_out)
            value = self.value_head(z).squeeze(-1)
            return value

    def evaluate_actions(
        self,
        x: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute action log probs, values and entropy for PPO."""
        lstm_out, new_hidden = self.lstm(x, hidden)
        last_out = lstm_out[:, -1, :]
        z = self.shared_mlp(last_out)
        
        logits = self.policy_head(z)
        value = self.value_head(z).squeeze(-1)
        
        dist = Categorical(logits=logits)
        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        return action_log_probs, value, dist_entropy

    def act(
        self,
        x: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass returning sampled action, log prob, value and new hidden state."""
        with torch.no_grad():
            lstm_out, new_hidden = self.lstm(x, hidden)
            last_out = lstm_out[:, -1, :]
            z = self.shared_mlp(last_out)
            logits = self.policy_head(z)
            value = self.value_head(z).squeeze(-1)
            
            dist = Categorical(logits=logits)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)
            
            return action, action_log_prob, value, new_hidden

    def init_hidden(self, batch_size: int, device: torch.device = torch.device("cpu")):
        h0 = torch.zeros(self.num_lstm_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros_like(h0)
        return h0, c0


def get_valid_actions(state, idx: int, player_idx: int) -> List[int]:
    """Return the list of valid actions for the specified player."""
    shuffled_hands = state._hands[idx]
    shuffled_reds = state._n_red_in_hands[idx]
    original_hands = shuffled_to_normal(state, shuffled_hands, idx)
    original_reds = shuffled_to_normal(state, shuffled_reds, idx)

    legal_action_list = state.legal_action_mask[idx].tolist()
    valid_indices = [i for i, val in enumerate(legal_action_list) if val]

    player_hand = original_hands[player_idx].tolist()
    red_tile_counts = original_reds[player_idx].tolist()
    complete_actions: List[int] = []

    for action in valid_indices:
        tile_count = player_hand[action]
        red_count = red_tile_counts[action]

        for _ in range(red_count):
            complete_actions.append(-action)
        for _ in range(tile_count - red_count):
            complete_actions.append(action)

    # Ensure we always have exactly VALID_ACTIONS_COUNT actions
    if len(complete_actions) != VALID_ACTIONS_COUNT:
        while len(complete_actions) < VALID_ACTIONS_COUNT:
            complete_actions.append(complete_actions[-1])
        complete_actions = complete_actions[:VALID_ACTIONS_COUNT]
    return complete_actions


def handle_opponent_moves(state, picked_actions: np.ndarray):
    """Fill in actions for non-player-1 agents using the heuristic bot."""
    active_games = ~(state.terminated | state.truncated)
    indices_other = np.where(active_games & (state.current_player != 0))[0]
    for idx in indices_other:
        current_player = state.current_player[idx]
        actions = get_valid_actions(state, idx, current_player)
        hand = actions_to_hand(actions)
        current_dora = state._dora[idx].tolist() + 1
        picked_tile = choose_next_move(hand, current_dora)
        picked_action = tile_to_action(picked_tile)
        picked_actions[idx] = picked_action
    return picked_actions


def discount_cumsum(x: List[float], gamma: float) -> List[float]:
    """Compute discounted cumulative sums of vectors."""
    out = [0.0] * len(x)
    running = 0.0
    for t in reversed(range(len(x))):
        running = x[t] + gamma * running
        out[t] = running
    return out


def normalize(x: torch.Tensor) -> torch.Tensor:
    """Normalize tensor with running statistics."""
    if len(x) == 0:
        return x
    mean = x.mean()
    std = x.std() + ADVANTAGE_EPSILON
    return (x - mean) / std


def ppo_update(
    model: ActorCriticLSTM,
    optimizer: torch.optim.Optimizer,
    trajectories: Dict[int, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]],
    episode_reward: np.ndarray,
) -> Tuple[float, float, float]:
    """Run PPO update on collected trajectories."""
    device = next(model.parameters()).device
    
    # Collect and process trajectory data
    states_all = []
    actions_all = []
    old_log_probs_all = []
    old_values_all = []
    returns_all = []
    advantages_all = []
    
    for env_idx, transitions in trajectories.items():
        if len(transitions) == 0:
            continue
            
        states, actions, log_probs, values, hiddens = [], [], [], [], []
        for s, a, lp, v, h in transitions:
            states.append(s)
            actions.append(a)
            log_probs.append(lp)
            values.append(v)
            
        # Stack episode data
        states = torch.stack(states)
        actions = torch.stack(actions)
        old_log_probs = torch.stack(log_probs)
        old_values = torch.stack(values)
        
        # Compute returns and advantages
        final_r = episode_reward[env_idx]
        disc_returns = torch.tensor(
            discount_cumsum([0.0] * (len(transitions) - 1) + [final_r], GAMMA),
            dtype=torch.float32,
            device=device
        )
        advantages = disc_returns - old_values.detach()
        
        # Collect for all episodes
        states_all.append(states)
        actions_all.append(actions)
        old_log_probs_all.append(old_log_probs)
        old_values_all.append(old_values)
        returns_all.append(disc_returns)
        advantages_all.append(advantages)
    
    if not states_all:  # No complete episodes
        return 0.0, 0.0, 0.0
        
    # Stack all episodes
    states = torch.cat(states_all)
    actions = torch.cat(actions_all)
    old_log_probs = torch.cat(old_log_probs_all)
    old_values = torch.cat(old_values_all)
    returns = torch.cat(returns_all)
    advantages = torch.cat(advantages_all)
    advantages = normalize(advantages)
    
    # Create dummy hidden states for evaluation (we don't need the actual sequence)
    batch_size = states.shape[0]
    dummy_hidden = model.init_hidden(batch_size, device)
    
    # PPO epochs
    policy_loss_epoch = 0
    value_loss_epoch = 0
    entropy_epoch = 0
    
    for _ in range(N_EPOCHS):
        # Get new log probs and values
        new_log_probs, new_values, entropy = model.evaluate_actions(
            states, 
            dummy_hidden,
            actions
        )
        
        # Policy loss with clipping
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss with clipping
        value_pred_clipped = old_values + (new_values - old_values).clamp(
            -VALUE_CLIP, VALUE_CLIP
        )
        value_losses = (new_values - returns).pow(2)
        value_losses_clipped = (value_pred_clipped - returns).pow(2)
        value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
        
        # Entropy bonus
        entropy_loss = -ENTROPY_COEF * entropy.mean()
        
        # Total loss and optimize
        loss = policy_loss + value_loss + entropy_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        
        policy_loss_epoch += policy_loss.item()
        value_loss_epoch += value_loss.item()
        entropy_epoch += entropy_loss.item()
    
    return (
        policy_loss_epoch / N_EPOCHS,
        value_loss_epoch / N_EPOCHS,
        entropy_epoch / N_EPOCHS
    )


def main():
    print(f"Running PPO training on {DEVICE} ...")

    # Environment init + JIT compile once
    init_fn, step_fn = init_env(ENV_ID)

    # Model + optimiser
    model = ActorCriticLSTM(
        input_size=INPUT_SIZE, 
        hidden_size=HIDDEN_SIZE, 
        num_actions=NUM_ACTIONS,
        num_lstm_layers=NUM_LSTM_LAYERS,
        num_fc_layers=NUM_FC_LAYERS,
        fc_hidden_size=FC_HIDDEN_SIZE
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # RNG setup
    KEY = reset_random_seed()

    # Training loop with progress bar
    with tqdm(total=MAX_UPDATES, desc="PPO Training", unit="update") as pbar:
        for update in range(1, MAX_UPDATES + 1):
            # ---- 1. rollout phase -------------------------------------------------
            KEY, subkey = jax.random.split(KEY)
            keys = jax.random.split(subkey, BATCH_SIZE)
            state = init_fn(keys)
            hidden = model.init_hidden(BATCH_SIZE, DEVICE)

            # Storage for trajectories (now grouped by environment)
            trajectories = defaultdict(list)  # env_id -> List[(state, action, log_prob, value, hidden)]
            done_flags = np.zeros(BATCH_SIZE, dtype=bool)
            episode_reward = np.zeros(BATCH_SIZE, dtype=np.float32)

            step_idx = 0
            while not done_flags.all():
                picked_actions = np.zeros(BATCH_SIZE, dtype=int)

                # Identify where our agent should act: (current_player == 0) & active
                active_games = ~(state.terminated | state.truncated)
                player_turns = (state.current_player == 0) & active_games
                indices = np.where(player_turns)[0]

                if len(indices) > 0:
                    inputs = []
                    valid_moves_batch: List[List[int]] = []
                    for idx in indices:
                        valid_moves = get_valid_actions(state, idx, 0)
                        valid_moves_batch.append(valid_moves)
                        input_vec = state_to_input(state, idx)
                        tensor_in = torch.tensor(input_vec, dtype=torch.float32).unsqueeze(0)
                        inputs.append(tensor_in)

                    input_batch = torch.stack(inputs, dim=0).to(DEVICE)
                    h0 = hidden[0][:, indices, :].contiguous()
                    c0 = hidden[1][:, indices, :].contiguous()
                    
                    # Get actions from policy
                    actions, log_probs, values, (h_n, c_n) = model.act(input_batch, (h0, c0))

                    for i, env_idx in enumerate(indices):
                        action_idx = actions[i].item()
                        picked_action = abs(valid_moves_batch[i][action_idx])
                        picked_actions[env_idx] = picked_action

                        # Store transition data for PPO
                        trajectories[env_idx].append((
                            input_batch[i],
                            actions[i],
                            log_probs[i],
                            values[i],
                            (h0[:, i:i+1, :].clone(), c0[:, i:i+1, :].clone())
                        ))

                    # Update hidden state
                    hidden[0][:, indices, :] = h_n
                    hidden[1][:, indices, :] = c_n

                # Rule-based opponents
                picked_actions = handle_opponent_moves(state, picked_actions)

                # Step env
                state = step_fn(state, picked_actions)

                # Capture reward deltas for finished episodes
                newly_done = (state.terminated | state.truncated) & (~done_flags)
                if newly_done.any():
                    finished_idx = np.where(newly_done)[0]
                    for idx in finished_idx:
                        r = float(state.rewards[idx, 0])
                        episode_reward[idx] = r
                    done_flags[newly_done] = True

                step_idx += 1
                if step_idx > MAX_EPISODE_STEPS:  # safety break
                    break

            # ---------- 2. PPO update ---------------------------------------------
            policy_loss, value_loss, entropy_loss = ppo_update(
                model, optimizer, trajectories, episode_reward
            )

            # Update progress bar
            total_ep_reward = episode_reward.sum()
            pbar.set_postfix({
                'reward': f'{total_ep_reward:.1f}',
                'p_loss': f'{policy_loss:.2f}',
                'v_loss': f'{value_loss:.2f}'
            })
            pbar.update(1)

            # periodic checkpoint
            if update % CHECKPOINT_FREQ == 0 or update == 1750:
                os.makedirs("checkpoints", exist_ok=True)
                ckpt_path = f"checkpoints/ppo_agent_update_{update}.pth"
                torch.save(model.state_dict(), ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main() 