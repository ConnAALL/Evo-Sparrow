"""
File that manages the game environment and provides utility functions for the training process.
"""

import os, random, sys, jax, torch
import numpy as np

# The upper-level file has the pgx module, so we have to add this to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pgx


# Tile names for the game
TILE_NAMES = {
    0: '1 Bamboo',
    1: '2 Bamboo',
    2: '3 Bamboo',
    3: '4 Bamboo',
    4: '5 Bamboo',
    5: '6 Bamboo',
    6: '7 Bamboo',
    7: '8 Bamboo',
    8: '9 Bamboo',
    9: 'Green Dragon',
    10: 'Red Dragon'
}


def init_env(env_id):
    """Initialize the game environment and its functions."""
    env = pgx.make(env_id)
    init = jax.jit(jax.vmap(env.init))
    step = jax.jit(jax.vmap(env.step))
    return init, step


def reset_random_seed():
    """Reset the random seed for reproducibility."""
    seed = random.randint(1000, 9999)
    return jax.random.PRNGKey(seed)


def array_to_tile_name(tiles):
    """Convert an array of tiles indices into a tile name."""
    for tile in tiles:
        tile = TILE_NAMES[tile]
        print(tile)


def array_to_tensor(array):
    """Convert a 1D array to a PyTorch"""
    tensor = torch.tensor(array, dtype=torch.float32)
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, INPUT_SIZE)
    return tensor


def shuffled_to_normal(state, array, game_id):
    """
    As the game state is shuffled (to prevent turn order bias),
    this function reorders the array to the original order.
    """
    shuffled_players = state._shuffled_players[game_id]
    reordered_array = [None] * 3
    for i in range(3): reordered_array[shuffled_players[i]] = array[i]
    return reordered_array


def state_to_input(state, idx):
    """
    Function to parse the player hand, discarded tiles, 
    and dora from the game state for a specific game index 
    in the batch to create the input for the LSTM."""

    # Parsing the player hand for the game at index idx
    shuffled_hands = state._hands[idx]
    shuffled_reds = state._n_red_in_hands[idx]
    shuffled_discards = state._rivers[idx]
    shuffled_red_discards = state._is_red_in_river[idx]

    # Undoing the shuffling
    original_hands = shuffled_to_normal(state, shuffled_hands, idx)
    original_reds = shuffled_to_normal(state, shuffled_reds, idx)
    original_discards = shuffled_to_normal(state, shuffled_discards, idx)
    original_red_discards = shuffled_to_normal(state, shuffled_red_discards, idx)

    player_1_hand = original_hands[0].tolist()
    player_1_reds = original_reds[0].tolist()
    representation = []
    for i in range(len(player_1_hand)):
        tile_count = player_1_hand[i]
        red_count = player_1_reds[i]
        normal_tile = tile_count - red_count
        for _ in range(red_count):
            representation.append(-1 * (i + 1))
        for _ in range(normal_tile):
            representation.append(i + 1)

    # Parsing the discarded tiles
    discarded_tiles = np.array(original_discards).flatten().tolist()
    red_discarded = np.array(original_red_discards).flatten().tolist()
    discarded = []
    for i in range(len(discarded_tiles)):
        tile = discarded_tiles[i]
        is_red = red_discarded[i]
        if tile == -1:
            discarded.append(0)
        else:
            discarded.append((tile + 1) * (-1 if is_red else 1))
    representation.extend(discarded)

    # Parsing the dora
    dora = state._dora[idx].item()
    representation.append(dora)
    return representation


def choose_next_move(hand, current_dora):
    """
    Rule-Based function to choose the next move of the agent.
    The agent will choose the tile with the highest discard priority score.
    """
    # First, count the occurrences of each tile in the hand
    tile_counts = {}
    for tile in hand:
        tile_value = abs(tile)
        tile_counts[tile_value] = tile_counts.get(tile_value, 0) + 1

    # Initialize a dictionary to store discard priority scores for each tile
    tile_scores = {}
    for tile in hand:
        tile_value = abs(tile)
        score = 0
        if tile < 0:
            score -= 5
        if tile_value == current_dora:
            score -= 5
        count = tile_counts[tile_value]
        if count >= 2:
            score -= 3
        adjacent_tiles = [tile_value - 2, tile_value - 1, tile_value + 1, tile_value + 2]
        has_adjacent = any(adj in tile_counts for adj in adjacent_tiles)
        if has_adjacent:
            score -= 2  # Tiles that can form sequences are valuable

        # Rule 5: Discard isolated tiles
        if not has_adjacent and count == 1:
            score += 2  # Isolated tiles are less useful

        # Rule 6: Discard edge tiles (1 and 9)
        if tile_value % 9 == 1 or tile_value % 9 == 9:
            score += 1  # Edge tiles have fewer sequence possibilities

        # Rule 7: Prefer middle tiles (3-7)
        if 3 <= tile_value % 9 <= 7:
            score -= 1  # Middle tiles are flexible for sequences

        # Rule 8: Discard honor tiles if not part of a set
        if tile_value > 27 and count < 2:
            score += 3  # Honor tiles are harder to use unless paired or triplet

        # Rule 9: Consider tile duplication
        if count == 1:
            score += 1  # Single tiles are less useful than multiples

        # Rule 10: Avoid breaking up completed sets
        if count >= 3:
            score -= 4
        tile_scores[tile] = score
    tile_to_discard = max(tile_scores, key=tile_scores.get)

    return tile_to_discard


def actions_to_hand(actions):
    """Convert action indices to hand tile representations, considering red tiles."""
    hand = []
    for action in actions:
        if action >= 0:
            hand.append(action + 1)  # Tiles are 1-indexed
        else:
            hand.append(action - 1)  # Negative indices for red tiles
    return hand


def tile_to_action(tile):
    """Convert a tile back to the action index, considering red tiles."""
    if tile >= 0:
        return tile - 1  # Convert back to 0-indexed action
    else:
        return -1 * (tile + 1)  # Adjust for negative red tile indices
