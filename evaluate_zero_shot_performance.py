"""
Phase 6A: Zero-shot Performance Evaluation for Hex DQN/ConvDQN agent.

This script evaluates a trained 7x7 model under:
1. Different board sizes
2. Unseen starting positions
3. Opponent variations

Important:
The trained model has a fixed 7x7 action space with 49 outputs.
Therefore, direct transfer to other board sizes is not supported.
However, this script also runs an experimental board-size transfer by projecting
other board sizes into the trained 7x7 input space:

    smaller boards -> centered/padded into 7x7
    larger boards  -> center crop into 7x7

This is an exploratory zero-shot transfer experiment, not a true size-invariant model.

Opponent setup:
Only Random Agent is used.
Opponent variations are implemented as Random Agent variants with different seeds.

Outputs:
    results/phase6_zero_shot_board_size_compatibility.csv
    results/phase6_zero_shot_experimental_board_size_transfer_results.csv
    results/phase6_zero_shot_experimental_board_size_transfer_summary.csv
    results/phase6_zero_shot_starting_positions_results.csv
    results/phase6_zero_shot_starting_positions_summary.csv
    results/phase6_zero_shot_random_seed_variations_results.csv
    results/phase6_zero_shot_random_seed_variations_summary.csv
    results/phase6_zero_shot_summary.csv

    results/phase6_zero_shot_board_size_compatibility.png
    results/phase6_zero_shot_experimental_board_size_transfer.png
    results/phase6_zero_shot_starting_positions.png
    results/phase6_zero_shot_random_seed_variations.png
"""

import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch

from hex_engine import hexPosition, RED, BLUE, EMPTY
from models import DQN, ConvDQN, board_to_tensor, board_to_spatial_tensor


# ============================================================
# Default configuration
# ============================================================

TRAINED_BOARD_SIZE = 7
TRAINED_ACTION_SPACE = TRAINED_BOARD_SIZE * TRAINED_BOARD_SIZE
RESULTS_DIR = "results"

DEFAULT_FINAL_MODEL_PATH = "results/phase5_cnn_reward_shaping_soft_path_block_7x7.pt"
DEFAULT_BEST_MODEL_PATH = "results/phase5_cnn_reward_shaping_soft_path_block_7x7_best.pt"

FALLBACK_MODEL_PATHS = [
    "results/phase5_cnn_reward_shaping_soft_path_block_7x7.pt",
    "results/phase5_cnn_reward_shaping_soft_path_block_7x7_best.pt",
    "results/phase5_cnn_reward_shaping_path_block_7x7.pt",
    "results/phase5_cnn_reward_shaping_7x7.pt",
    "results/phase4_model_7x7.pt",
]


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Model loading
# ============================================================

def select_model_path(checkpoint_type, explicit_model_path):
    if explicit_model_path is not None:
        if not os.path.exists(explicit_model_path):
            raise FileNotFoundError(f"Explicit model path not found: {explicit_model_path}")
        return explicit_model_path

    if checkpoint_type == "final" and os.path.exists(DEFAULT_FINAL_MODEL_PATH):
        return DEFAULT_FINAL_MODEL_PATH

    if checkpoint_type == "best" and os.path.exists(DEFAULT_BEST_MODEL_PATH):
        return DEFAULT_BEST_MODEL_PATH

    for path in FALLBACK_MODEL_PATHS:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "No trained model checkpoint found. Expected one of:\n"
        + "\n".join(FALLBACK_MODEL_PATHS)
    )


def extract_checkpoint_data(checkpoint):
    """
    Supports:
        torch.save(model.state_dict(), path)
        torch.save({"model_state_dict": ..., "board_size": ..., "use_cnn": ...}, path)
    """
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        metadata = checkpoint
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        metadata = {}
    else:
        raise RuntimeError("Unsupported checkpoint format.")

    return state_dict, metadata


def infer_architecture_from_state_dict(state_dict, metadata):
    if isinstance(metadata, dict) and "use_cnn" in metadata:
        return bool(metadata["use_cnn"])

    keys = list(state_dict.keys())

    if any(key.startswith("conv.") for key in keys):
        return True

    if any(key.startswith("network.") for key in keys):
        return False

    raise RuntimeError(
        "Could not infer architecture from checkpoint. "
        "Expected ConvDQN keys starting with 'conv.' or DQN keys starting with 'network.'."
    )


def load_checkpoint_model(model_path, device):
    """
    Always loads the trained model as a 7x7 model.
    This is necessary because the checkpoint output layer has 49 actions.
    """
    checkpoint = torch.load(model_path, map_location=device)
    state_dict, metadata = extract_checkpoint_data(checkpoint)

    checkpoint_board_size = metadata.get("board_size", TRAINED_BOARD_SIZE)

    if checkpoint_board_size != TRAINED_BOARD_SIZE:
        raise RuntimeError(
            f"Checkpoint board size is {checkpoint_board_size}, "
            f"but this zero-shot script expects a trained 7x7 model."
        )

    use_cnn = infer_architecture_from_state_dict(state_dict, metadata)

    if use_cnn:
        model = ConvDQN(board_size=TRAINED_BOARD_SIZE).to(device)
        model_class = "ConvDQN"
    else:
        model = DQN(board_size=TRAINED_BOARD_SIZE).to(device)
        model_class = "DQN"

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model, use_cnn, model_class, metadata


# ============================================================
# Canonical board logic
# Same logic as Phase 4 / Phase 5 evaluation
# ============================================================

def transform_move_for_blue(move, board_size):
    row, col = move
    return board_size - 1 - col, board_size - 1 - row


def canonical_board(board, player):
    size = len(board)

    if player == RED:
        return board

    transformed = [[EMPTY for _ in range(size)] for _ in range(size)]

    for row in range(size):
        for col in range(size):
            value = board[size - 1 - col][size - 1 - row]

            if value == RED:
                transformed[row][col] = BLUE
            elif value == BLUE:
                transformed[row][col] = RED
            else:
                transformed[row][col] = EMPTY

    return transformed


def canonical_action_set(action_set, player, board_size):
    if player == RED:
        return action_set

    return [transform_move_for_blue(move, board_size) for move in action_set]


def inverse_canonical_move(move, player, board_size):
    if player == RED:
        return move

    return transform_move_for_blue(move, board_size)


# ============================================================
# Projection between arbitrary board size and trained 7x7 space
# ============================================================

def project_board_to_7x7(board):
    """
    Project an arbitrary board into 7x7 trained input space.

    If board is smaller than 7x7:
        place it into the center/top-left-centered part of a 7x7 empty board.

    If board is larger than 7x7:
        use a center crop.

    Returns:
        projected_board,
        metadata dict with mode and offsets.
    """
    source_size = len(board)

    projected = [[EMPTY for _ in range(TRAINED_BOARD_SIZE)] for _ in range(TRAINED_BOARD_SIZE)]

    if source_size == TRAINED_BOARD_SIZE:
        return board, {
            "mode": "native_7x7",
            "source_size": source_size,
            "row_offset": 0,
            "col_offset": 0,
        }

    if source_size < TRAINED_BOARD_SIZE:
        row_offset = (TRAINED_BOARD_SIZE - source_size) // 2
        col_offset = (TRAINED_BOARD_SIZE - source_size) // 2

        for row in range(source_size):
            for col in range(source_size):
                projected[row + row_offset][col + col_offset] = board[row][col]

        return projected, {
            "mode": "pad_to_7x7",
            "source_size": source_size,
            "row_offset": row_offset,
            "col_offset": col_offset,
        }

    row_offset = (source_size - TRAINED_BOARD_SIZE) // 2
    col_offset = (source_size - TRAINED_BOARD_SIZE) // 2

    for row in range(TRAINED_BOARD_SIZE):
        for col in range(TRAINED_BOARD_SIZE):
            projected[row][col] = board[row + row_offset][col + col_offset]

    return projected, {
        "mode": "center_crop_to_7x7",
        "source_size": source_size,
        "row_offset": row_offset,
        "col_offset": col_offset,
    }


def project_move_to_7x7(move, projection_info):
    """
    Map a move from source board coordinates into projected 7x7 coordinates.

    Returns None if the move is outside the 7x7 crop for larger boards.
    """
    row, col = move
    source_size = projection_info["source_size"]
    row_offset = projection_info["row_offset"]
    col_offset = projection_info["col_offset"]
    mode = projection_info["mode"]

    if mode == "native_7x7":
        return move

    if source_size < TRAINED_BOARD_SIZE:
        return row + row_offset, col + col_offset

    projected_row = row - row_offset
    projected_col = col - col_offset

    if 0 <= projected_row < TRAINED_BOARD_SIZE and 0 <= projected_col < TRAINED_BOARD_SIZE:
        return projected_row, projected_col

    return None


def inverse_project_move_from_7x7(projected_move, projection_info):
    """
    Map a selected 7x7 projected move back to source board coordinates.
    """
    row, col = projected_move
    source_size = projection_info["source_size"]
    row_offset = projection_info["row_offset"]
    col_offset = projection_info["col_offset"]
    mode = projection_info["mode"]

    if mode == "native_7x7":
        return projected_move

    if source_size < TRAINED_BOARD_SIZE:
        return row - row_offset, col - col_offset

    return row + row_offset, col + col_offset


# ============================================================
# DQN move selection
# ============================================================

def dqn_select_move_native_7x7(board, action_set, player, model, use_cnn, device):
    """
    Standard selection for native 7x7 evaluation.
    """
    if not action_set:
        return None

    board_size = TRAINED_BOARD_SIZE

    state = canonical_board(board, player)
    valid_actions = canonical_action_set(action_set, player, board_size)

    with torch.no_grad():
        if use_cnn:
            state_tensor = board_to_spatial_tensor(state, device=device)
        else:
            state_tensor = board_to_tensor(state, device=device)

        q_values = model(state_tensor).squeeze(0)

    best_move = None
    best_value = -float("inf")

    for move in valid_actions:
        row, col = move
        action_index = row * TRAINED_BOARD_SIZE + col
        value = q_values[action_index].item()

        if value > best_value:
            best_value = value
            best_move = move

    real_move = inverse_canonical_move(best_move, player, board_size)

    if real_move not in action_set:
        return random.choice(action_set)

    return real_move


def dqn_select_move_experimental_transfer(board, action_set, player, model, use_cnn, device):
    """
    Experimental move selection for arbitrary board sizes.

    Steps:
        1. Convert board to canonical current-player perspective.
        2. Project/crop/pad canonical board into 7x7.
        3. Project valid canonical actions into 7x7.
        4. Select best Q-value among projected valid actions.
        5. Map selected action back to source board.
        6. Invert canonical move back to real board orientation.

    For boards larger than 7x7, moves outside the center crop cannot be selected.
    If no valid move lies in the projected 7x7 crop, we fall back to random.
    """
    if not action_set:
        return None

    source_board_size = len(board)

    if source_board_size == TRAINED_BOARD_SIZE:
        return dqn_select_move_native_7x7(
            board=board,
            action_set=action_set,
            player=player,
            model=model,
            use_cnn=use_cnn,
            device=device,
        )

    canonical = canonical_board(board, player)
    canonical_actions = canonical_action_set(action_set, player, source_board_size)

    projected_board, projection_info = project_board_to_7x7(canonical)

    projected_action_pairs = []

    for canonical_move in canonical_actions:
        projected_move = project_move_to_7x7(canonical_move, projection_info)

        if projected_move is not None:
            pr, pc = projected_move

            if 0 <= pr < TRAINED_BOARD_SIZE and 0 <= pc < TRAINED_BOARD_SIZE:
                projected_action_pairs.append((canonical_move, projected_move))

    if not projected_action_pairs:
        return random.choice(action_set)

    with torch.no_grad():
        if use_cnn:
            state_tensor = board_to_spatial_tensor(projected_board, device=device)
        else:
            state_tensor = board_to_tensor(projected_board, device=device)

        q_values = model(state_tensor).squeeze(0)

    best_canonical_move = None
    best_value = -float("inf")

    for canonical_move, projected_move in projected_action_pairs:
        pr, pc = projected_move
        action_index = pr * TRAINED_BOARD_SIZE + pc
        value = q_values[action_index].item()

        if value > best_value:
            best_value = value
            best_canonical_move = canonical_move

    real_move = inverse_canonical_move(best_canonical_move, player, source_board_size)

    if real_move not in action_set:
        return random.choice(action_set)

    return real_move


# ============================================================
# Random opponent only
# ============================================================

class SeededRandomOpponent:
    """
    Random Agent variant with its own seed.
    Used for opponent variations without introducing other agent types.
    """

    def __init__(self, seed: int):
        self.seed = seed
        self.name = f"Random Agent seed {seed}"
        self.rng = random.Random(seed)

    def select_move(self, board, action_set):
        if not action_set:
            return None

        return self.rng.choice(action_set)


def random_opponent_move(board, action_set):
    if not action_set:
        return None

    return random.choice(action_set)


# ============================================================
# Starting positions
# ============================================================

def apply_unseen_starting_position(game, num_prefilled_stones, seed):
    """
    Apply random legal moves before evaluation starts.
    Uses hexPosition.move(), so player switching and winner detection stay consistent
    with the original game engine.
    """
    rng = random.Random(seed)
    applied_moves = []

    for _ in range(num_prefilled_stones):
        if game.winner != EMPTY:
            break

        action_set = game.get_action_space()

        if not action_set:
            break

        move = rng.choice(action_set)
        game.move(move)
        applied_moves.append(move)

    return applied_moves


# ============================================================
# Game simulation
# ============================================================

def count_stones(board):
    return sum(1 for row in board for value in row if value != EMPTY)


def play_game(
    dqn_color,
    model,
    use_cnn,
    board_size,
    device,
    start_stones=0,
    seed=42,
    opponent=None,
    experimental_board_transfer=False,
):
    game = hexPosition(size=board_size)

    apply_unseen_starting_position(
        game=game,
        num_prefilled_stones=start_stones,
        seed=seed,
    )

    invalid_fallbacks = 0

    while game.winner == EMPTY:
        action_set = game.get_action_space()

        if not action_set:
            break

        current_player = game.player

        if current_player == dqn_color:
            if experimental_board_transfer:
                move = dqn_select_move_experimental_transfer(
                    board=game.board,
                    action_set=action_set,
                    player=current_player,
                    model=model,
                    use_cnn=use_cnn,
                    device=device,
                )
            else:
                move = dqn_select_move_native_7x7(
                    board=game.board,
                    action_set=action_set,
                    player=current_player,
                    model=model,
                    use_cnn=use_cnn,
                    device=device,
                )
        else:
            if opponent is None:
                move = random_opponent_move(game.board, action_set)
            else:
                move = opponent.select_move(game.board, action_set)

        if move not in action_set:
            invalid_fallbacks += 1
            move = random.choice(action_set)

        game.move(move)

    winner = game.winner
    dqn_win = int(winner == dqn_color)

    return {
        "winner": winner,
        "dqn_color": dqn_color,
        "dqn_win": dqn_win,
        "game_length": count_stones(game.board),
        "start_stones": start_stones,
        "invalid_fallbacks": invalid_fallbacks,
    }


def evaluate_setting(
    experiment,
    model,
    use_cnn,
    board_size,
    device,
    episodes,
    start_stones,
    seed,
    opponent_name,
    opponent_factory=None,
    experimental_board_transfer=False,
):
    rows = []

    opponent = opponent_factory() if opponent_factory is not None else None

    for game_id in range(episodes):
        dqn_color = RED if game_id % 2 == 0 else BLUE
        episode_seed = seed + game_id + start_stones * 10_000 + board_size * 100_000

        result = play_game(
            dqn_color=dqn_color,
            model=model,
            use_cnn=use_cnn,
            board_size=board_size,
            device=device,
            start_stones=start_stones,
            seed=episode_seed,
            opponent=opponent,
            experimental_board_transfer=experimental_board_transfer,
        )

        rows.append(
            {
                "experiment": experiment,
                "episode": game_id + 1,
                "board_size": board_size,
                "start_stones": start_stones,
                "opponent": opponent_name,
                "dqn_color": "RED" if result["dqn_color"] == RED else "BLUE",
                "winner": "RED" if result["winner"] == RED else "BLUE",
                "dqn_win": result["dqn_win"],
                "game_length": result["game_length"],
                "invalid_fallbacks": result["invalid_fallbacks"],
                "experimental_board_transfer": experimental_board_transfer,
            }
        )

    return pd.DataFrame(rows)


def summarize_results(df):
    summary = (
        df.groupby(
            [
                "experiment",
                "board_size",
                "start_stones",
                "opponent",
                "experimental_board_transfer",
            ],
            as_index=False,
        )
        .agg(
            games=("dqn_win", "count"),
            dqn_wins=("dqn_win", "sum"),
            win_rate=("dqn_win", "mean"),
            avg_game_length=("game_length", "mean"),
            std_game_length=("game_length", "std"),
            invalid_fallbacks=("invalid_fallbacks", "sum"),
        )
    )

    red_summary = (
        df[df["dqn_color"] == "RED"]
        .groupby(
            [
                "experiment",
                "board_size",
                "start_stones",
                "opponent",
                "experimental_board_transfer",
            ],
            as_index=False,
        )
        .agg(
            red_games=("dqn_win", "count"),
            dqn_red_wins=("dqn_win", "sum"),
            red_win_rate=("dqn_win", "mean"),
        )
    )

    blue_summary = (
        df[df["dqn_color"] == "BLUE"]
        .groupby(
            [
                "experiment",
                "board_size",
                "start_stones",
                "opponent",
                "experimental_board_transfer",
            ],
            as_index=False,
        )
        .agg(
            blue_games=("dqn_win", "count"),
            dqn_blue_wins=("dqn_win", "sum"),
            blue_win_rate=("dqn_win", "mean"),
        )
    )

    summary = summary.merge(
        red_summary,
        on=[
            "experiment",
            "board_size",
            "start_stones",
            "opponent",
            "experimental_board_transfer",
        ],
        how="left",
    )

    summary = summary.merge(
        blue_summary,
        on=[
            "experiment",
            "board_size",
            "start_stones",
            "opponent",
            "experimental_board_transfer",
        ],
        how="left",
    )

    summary["dqn_losses"] = summary["games"] - summary["dqn_wins"]

    ordered_cols = [
        "experiment",
        "board_size",
        "start_stones",
        "opponent",
        "experimental_board_transfer",
        "games",
        "dqn_wins",
        "dqn_losses",
        "win_rate",
        "red_games",
        "dqn_red_wins",
        "red_win_rate",
        "blue_games",
        "dqn_blue_wins",
        "blue_win_rate",
        "avg_game_length",
        "std_game_length",
        "invalid_fallbacks",
    ]

    return summary[ordered_cols]


# ============================================================
# Board-size compatibility
# ============================================================

def evaluate_board_size_compatibility(board_sizes, model_path, model_class):
    rows = []

    for tested_size in board_sizes:
        required_action_space = tested_size * tested_size

        if tested_size == TRAINED_BOARD_SIZE:
            status = "direct_supported"
            reason = "matches_trained_7x7_action_space"
        else:
            status = "direct_unsupported_but_experimental_transfer_used"
            reason = (
                "fixed_7x7_action_space; direct zero-shot transfer is unsupported, "
                "but experimental projection/cropping into 7x7 is evaluated separately"
            )

        rows.append(
            {
                "model_path": model_path,
                "model_class": model_class,
                "trained_board_size": TRAINED_BOARD_SIZE,
                "tested_board_size": tested_size,
                "model_action_space": TRAINED_ACTION_SPACE,
                "required_action_space": required_action_space,
                "direct_transfer_status": status,
                "reason": reason,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Plotting
# ============================================================

def plot_board_size_compatibility(df, output_dir):
    plot_df = df.copy()
    plot_df["direct_compatible"] = plot_df["direct_transfer_status"].map(
        {
            "direct_supported": 1,
            "direct_unsupported_but_experimental_transfer_used": 0,
        }
    )

    plt.figure(figsize=(8, 5))
    plt.bar(plot_df["tested_board_size"].astype(str), plot_df["direct_compatible"])
    plt.ylim(0, 1.2)
    plt.yticks([0, 1], ["Direct unsupported", "Direct supported"])
    plt.xlabel("Tested board size")
    plt.ylabel("Direct compatibility")
    plt.title("Phase 6A: Direct Board Size Compatibility")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "phase6_zero_shot_board_size_compatibility.png"),
        dpi=150,
    )
    plt.close()


def plot_experimental_board_size_transfer(summary_df, output_dir):
    plot_df = summary_df.sort_values("board_size")

    plt.figure(figsize=(8, 5))
    plt.plot(plot_df["board_size"], plot_df["win_rate"], marker="o")
    plt.ylim(0, 1)
    plt.xlabel("Board size")
    plt.ylabel("DQN win rate")
    plt.title("Phase 6A: Experimental Board-Size Transfer via 7x7 Projection")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "phase6_zero_shot_experimental_board_size_transfer.png"),
        dpi=150,
    )
    plt.close()


def plot_starting_positions(summary_df, output_dir):
    plot_df = summary_df.sort_values("start_stones")

    plt.figure(figsize=(8, 5))
    plt.plot(plot_df["start_stones"], plot_df["win_rate"], marker="o")
    plt.ylim(0, 1)
    plt.xlabel("Number of random pre-filled stones")
    plt.ylabel("DQN win rate")
    plt.title("Phase 6A: Zero-shot Performance on Unseen Starting Positions")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "phase6_zero_shot_starting_positions.png"),
        dpi=150,
    )
    plt.close()


def plot_random_seed_variations(summary_df, output_dir):
    plot_df = summary_df.sort_values("opponent")

    plt.figure(figsize=(9, 5))
    plt.bar(plot_df["opponent"], plot_df["win_rate"])
    plt.ylim(0, 1)
    plt.xlabel("Random opponent variant")
    plt.ylabel("DQN win rate")
    plt.title("Phase 6A: Zero-shot Performance Against Random Seed Variations")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "phase6_zero_shot_random_seed_variations.png"),
        dpi=150,
    )
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 6A: Zero-shot performance evaluation for trained Hex DQN agents."
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional explicit model checkpoint path.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        choices=["final", "best"],
        default="final",
        help=(
            "Which Phase 5 checkpoint to use if --model-path is not provided. "
            "Default is 'final' because it matches the standard Phase 5 evaluation file."
        ),
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="Number of games per setting.",
    )

    parser.add_argument(
        "--board-sizes",
        type=int,
        nargs="+",
        default=[5, 6, 7, 8, 9],
        help="Board sizes for experimental transfer evaluation.",
    )

    parser.add_argument(
        "--start-stones",
        type=int,
        nargs="+",
        default=[0, 2, 4, 6, 8],
        help="Numbers of random pre-filled stones for unseen starting position tests.",
    )

    parser.add_argument(
        "--opponent-seeds",
        type=int,
        nargs="+",
        default=[42, 123, 2026, 9001],
        help="Seeds used to create Random Agent opponent variations.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed.",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model_path = select_model_path(
        checkpoint_type=args.checkpoint,
        explicit_model_path=args.model_path,
    )

    model, use_cnn, model_class, metadata = load_checkpoint_model(
        model_path=model_path,
        device=device,
    )

    print("=" * 80)
    print("Phase 6A: Zero-shot Performance Evaluation")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Selected model: {model_path}")
    print(f"Model class: {model_class}")
    print(f"Checkpoint type: {args.checkpoint}")
    print(f"Trained board size: {TRAINED_BOARD_SIZE}x{TRAINED_BOARD_SIZE}")
    print(f"Model action space: {TRAINED_ACTION_SPACE}")
    print(f"Episodes per setting: {args.episodes}")
    print("Opponent setup: Random Agent only")
    print("Board-size transfer: experimental 7x7 projection/crop/padding")
    print("=" * 80)

    # ------------------------------------------------------------
    # 1A. Direct board-size compatibility check
    # ------------------------------------------------------------
    board_compatibility_df = evaluate_board_size_compatibility(
        board_sizes=args.board_sizes,
        model_path=model_path,
        model_class=model_class,
    )

    board_compatibility_path = os.path.join(
        RESULTS_DIR,
        "phase6_zero_shot_board_size_compatibility.csv",
    )

    board_compatibility_df.to_csv(board_compatibility_path, index=False)
    plot_board_size_compatibility(board_compatibility_df, RESULTS_DIR)

    # ------------------------------------------------------------
    # 1B. Experimental board-size transfer against Random Agent
    # ------------------------------------------------------------
    board_size_transfer_results = []

    for board_size in args.board_sizes:
        result_df = evaluate_setting(
            experiment="experimental_board_size_transfer",
            model=model,
            use_cnn=use_cnn,
            board_size=board_size,
            device=device,
            episodes=args.episodes,
            start_stones=0,
            seed=args.seed,
            opponent_name="Random Agent",
            opponent_factory=None,
            experimental_board_transfer=True,
        )

        board_size_transfer_results.append(result_df)

    board_size_transfer_results_df = pd.concat(
        board_size_transfer_results,
        ignore_index=True,
    )

    board_size_transfer_summary_df = summarize_results(board_size_transfer_results_df)

    board_size_transfer_results_path = os.path.join(
        RESULTS_DIR,
        "phase6_zero_shot_experimental_board_size_transfer_results.csv",
    )

    board_size_transfer_summary_path = os.path.join(
        RESULTS_DIR,
        "phase6_zero_shot_experimental_board_size_transfer_summary.csv",
    )

    board_size_transfer_results_df.to_csv(board_size_transfer_results_path, index=False)
    board_size_transfer_summary_df.to_csv(board_size_transfer_summary_path, index=False)
    plot_experimental_board_size_transfer(board_size_transfer_summary_df, RESULTS_DIR)

    # ------------------------------------------------------------
    # 2. Unseen starting positions on native 7x7 against Random Agent
    # ------------------------------------------------------------
    starting_position_results = []

    for start_stones in args.start_stones:
        result_df = evaluate_setting(
            experiment="unseen_starting_positions",
            model=model,
            use_cnn=use_cnn,
            board_size=TRAINED_BOARD_SIZE,
            device=device,
            episodes=args.episodes,
            start_stones=start_stones,
            seed=args.seed,
            opponent_name="Random Agent",
            opponent_factory=None,
            experimental_board_transfer=False,
        )

        starting_position_results.append(result_df)

    starting_position_results_df = pd.concat(
        starting_position_results,
        ignore_index=True,
    )

    starting_position_summary_df = summarize_results(starting_position_results_df)

    starting_results_path = os.path.join(
        RESULTS_DIR,
        "phase6_zero_shot_starting_positions_results.csv",
    )

    starting_summary_path = os.path.join(
        RESULTS_DIR,
        "phase6_zero_shot_starting_positions_summary.csv",
    )

    starting_position_results_df.to_csv(starting_results_path, index=False)
    starting_position_summary_df.to_csv(starting_summary_path, index=False)
    plot_starting_positions(starting_position_summary_df, RESULTS_DIR)

    # ------------------------------------------------------------
    # 3. Opponent variations: Random Agent with different seeds
    # ------------------------------------------------------------
    seed_variation_results = []

    for opponent_seed in args.opponent_seeds:
        result_df = evaluate_setting(
            experiment="random_seed_opponent_variations",
            model=model,
            use_cnn=use_cnn,
            board_size=TRAINED_BOARD_SIZE,
            device=device,
            episodes=args.episodes,
            start_stones=0,
            seed=args.seed,
            opponent_name=f"Random Agent seed {opponent_seed}",
            opponent_factory=lambda s=opponent_seed: SeededRandomOpponent(seed=s),
            experimental_board_transfer=False,
        )

        seed_variation_results.append(result_df)

    seed_variation_results_df = pd.concat(
        seed_variation_results,
        ignore_index=True,
    )

    seed_variation_summary_df = summarize_results(seed_variation_results_df)

    seed_variation_results_path = os.path.join(
        RESULTS_DIR,
        "phase6_zero_shot_random_seed_variations_results.csv",
    )

    seed_variation_summary_path = os.path.join(
        RESULTS_DIR,
        "phase6_zero_shot_random_seed_variations_summary.csv",
    )

    seed_variation_results_df.to_csv(seed_variation_results_path, index=False)
    seed_variation_summary_df.to_csv(seed_variation_summary_path, index=False)
    plot_random_seed_variations(seed_variation_summary_df, RESULTS_DIR)

    # ------------------------------------------------------------
    # Combined summary
    # ------------------------------------------------------------
    combined_summary_df = pd.concat(
        [
            board_size_transfer_summary_df,
            starting_position_summary_df,
            seed_variation_summary_df,
        ],
        ignore_index=True,
    )

    combined_summary_path = os.path.join(
        RESULTS_DIR,
        "phase6_zero_shot_summary.csv",
    )

    combined_summary_df.to_csv(combined_summary_path, index=False)

    print("\nDirect board size compatibility:")
    print(board_compatibility_df.to_string(index=False))

    print("\nExperimental board-size transfer summary:")
    print(board_size_transfer_summary_df.to_string(index=False))

    print("\nUnseen starting positions summary:")
    print(starting_position_summary_df.to_string(index=False))

    print("\nRandom seed opponent variations summary:")
    print(seed_variation_summary_df.to_string(index=False))

    print("\nSaved files:")
    print(f"  {board_compatibility_path}")
    print(f"  {board_size_transfer_results_path}")
    print(f"  {board_size_transfer_summary_path}")
    print(f"  {starting_results_path}")
    print(f"  {starting_summary_path}")
    print(f"  {seed_variation_results_path}")
    print(f"  {seed_variation_summary_path}")
    print(f"  {combined_summary_path}")
    print(f"  {os.path.join(RESULTS_DIR, 'phase6_zero_shot_board_size_compatibility.png')}")
    print(f"  {os.path.join(RESULTS_DIR, 'phase6_zero_shot_experimental_board_size_transfer.png')}")
    print(f"  {os.path.join(RESULTS_DIR, 'phase6_zero_shot_starting_positions.png')}")
    print(f"  {os.path.join(RESULTS_DIR, 'phase6_zero_shot_random_seed_variations.png')}")

    print("\nInterpretation note:")
    print(
        "The gameplay evaluation uses the original hex_engine.hexPosition and the same "
        "canonical RED/BLUE transformation used in the standard Phase 5 evaluation. "
        "The trained DQN/ConvDQN checkpoint has a fixed 7x7 output layer with 49 actions. "
        "Therefore, direct transfer to other board sizes is not supported. "
        "For Phase 6A, different board sizes are additionally evaluated with an experimental "
        "projection/crop/padding transfer into the trained 7x7 input space. "
        "Opponent variations are implemented only with Random Agent variants using different seeds."
    )
    print("=" * 80)


if __name__ == "__main__":
    main()