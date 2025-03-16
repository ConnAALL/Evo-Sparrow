import os, random, sys, jax, torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pgx


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


def shuffled_to_normal(state, array, game_id):
    shuffled_players = state._shuffled_players[game_id]
    reordered_array = [None] * 3
    for i in range(3): reordered_array[shuffled_players[i]] = array[i]
    return reordered_array


def state_to_input(state, idx, player_idx):
    """Parse the state for the given game index and player index to create a fixed-size input vector for the LSTM."""
    # Parsing the player's hand for the game at index idx
    shuffled_hands = state._hands[idx]
    shuffled_reds = state._n_red_in_hands[idx]
    shuffled_discards = state._rivers[idx]
    shuffled_red_discards = state._is_red_in_river[idx]

    # Undoing the shuffling
    original_hands = shuffled_to_normal(state, shuffled_hands, idx)
    original_reds = shuffled_to_normal(state, shuffled_reds, idx)
    original_discards = shuffled_to_normal(state, shuffled_discards, idx)
    original_red_discards = shuffled_to_normal(state, shuffled_red_discards, idx)

    # Use the appropriate player's hand
    player_hand = original_hands[player_idx].tolist()
    player_reds = original_reds[player_idx].tolist()
    representation = []
    for i in range(len(player_hand)):
        tile_count = player_hand[i]
        red_count = player_reds[i]
        normal_tile = tile_count - red_count
        for _ in range(red_count):
            representation.append(-1 * (i + 1))
        for _ in range(normal_tile):
            representation.append(i + 1)

    # Parsing discarded tiles (this might be shared among players, adjust if needed)
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

    # Ensure the output has exactly 37 elements (example: pad or truncate)
    expected_length = 37
    current_length = len(representation)
    if current_length < expected_length:
        # pad with zeros
        representation.extend([0] * (expected_length - current_length))
    elif current_length > expected_length:
        representation = representation[:expected_length]

    return representation


def choose_next_move(hand, current_dora):
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
