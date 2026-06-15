"""
Phase 6A: Zero-shot Performance Evaluation for Hex DQN/ConvDQN agent.

This script evaluates a trained 7x7 model under:
1. Different board sizes
2. Unseen starting positions
3. Opponent variations

Opponents:
    - Random Agent
    - Center/Greedy Agent
    - Random Agent variants with different seeds

Important:
The trained model has a fixed 7x7 action space with 49 outputs.
Therefore, direct transfer to other board sizes is not supported.

For board sizes different from 7x7, this script runs an experimental transfer:
    smaller boards -> centered/padded into 7x7
    larger boards  -> center crop into 7x7

This is an exploratory zero-shot transfer experiment, not a true size-invariant model.

Main outputs:
    results/phase6_zero_shot_results.csv
    results/phase6_zero_shot_summary.csv
    results/phase6_zero_shot_board_size_compatibility.csv

Plots:
    results/phase6_zero_shot_board_size_compatibility.png
    results/phase6_zero_shot_board_size_transfer.png
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
from agents import center_agent


# ============================================================
# Default configuration
# ============================================================

TRAINED_BOARD_SIZE = 7
TRAINED_ACTION_SPACE = TRAINED_BOARD_SIZE * TRAINED_BOARD_SIZE
RESULTS_DIR = "results"

DEFAULT_FINAL_MODEL_PATH = (
    "results/phase5_cnn_mixed_random_center_reward_shaping_soft_path_block_7x7.pt"
)

DEFAULT_BEST_MODEL_PATH = (
    "results/phase5_cnn_mixed_random_center_reward_shaping_soft_path_block_7x7_best.pt"
)

FALLBACK_MODEL_PATHS = [
    "results/phase5_cnn_mixed_random_center_reward_shaping_soft_path_block_7x7.pt",
    "results/phase5_cnn_mixed_random_center_reward_shaping_soft_path_block_7x7_best.pt",
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
    checkpoint = torch.load(model_path, map_location=device)
    state_dict, metadata = extract_checkpoint_data(checkpoint)

    checkpoint_board_size = metadata.get("board_size", TRAINED_BOARD_SIZE)

    if checkpoint_board_size != TRAINED_BOARD_SIZE:
        raise RuntimeError(
            f"Checkpoint board size is {checkpoint_board_size}, "
            f"but this script expects a trained 7x7 model."
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
    source_size = len(board)

    projected = [
        [EMPTY for _ in range(TRAINED_BOARD_SIZE)]
        for _ in range(TRAINED_BOARD_SIZE)
    ]

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


# ============================================================
# DQN move selection
# ============================================================

def dqn_select_move_native_7x7(board, action_set, player, model, use_cnn, device):
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

        if projected_move is None:
            continue

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

    real_move = inverse_canonical_move(
        best_canonical_move,
        player,
        source_board_size,
    )

    if real_move not in action_set:
        return random.choice(action_set)

    return real_move


# ============================================================
# Opponents
# ============================================================

class RandomOpponent:
    def __init__(self, seed=None, name="Random Agent"):
        self.name = name
        self.rng = random.Random(seed) if seed is not None else random

    def select_move(self, board, action_set):
        if not action_set:
            return None

        return self.rng.choice(action_set)


class CenterGreedyOpponent:
    def __init__(self):
        self.name = "Center/Greedy Agent"

    def select_move(self, board, action_set):
        if not action_set:
            return None

        return center_agent(board, action_set)


# ============================================================
# Starting positions
# ============================================================

def apply_unseen_starting_position(game, num_prefilled_stones, seed):
    rng = random.Random(seed)

    for _ in range(num_prefilled_stones):
        if game.winner != EMPTY:
            break

        action_set = game.get_action_space()

        if not action_set:
            break

        move = rng.choice(action_set)
        game.move(move)


# ============================================================
# Game simulation and evaluation
# ============================================================

def count_stones(board):
    return sum(1 for row in board for value in row if value != EMPTY)


def play_game(
    dqn_color,
    model,
    use_cnn,
    board_size,
    device,
    opponent,
    start_stones=0,
    seed=42,
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
    opponent_factory,
    experimental_board_transfer=False,
):
    rows = []
    opponent = opponent_factory()

    for game_id in range(episodes):
        dqn_color = RED if game_id % 2 == 0 else BLUE
        episode_seed = seed + game_id + start_stones * 10_000 + board_size * 100_000

        result = play_game(
            dqn_color=dqn_color,
            model=model,
            use_cnn=use_cnn,
            board_size=board_size,
            device=device,
            opponent=opponent,
            start_stones=start_stones,
            seed=episode_seed,
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

    path = os.path.join(RESULTS_DIR, "phase6_zero_shot_board_size_compatibility.png")
    plt.savefig(path, dpi=150)
    plt.close()


def plot_board_size_transfer(summary_df):
    plot_df = summary_df[
        summary_df["experiment"] == "experimental_board_size_transfer"
    ].sort_values(["opponent", "board_size"])

    if plot_df.empty:
        return

    plt.figure(figsize=(8, 5))

    for opponent_name in sorted(plot_df["opponent"].unique()):
        subset = plot_df[plot_df["opponent"] == opponent_name]

        plt.plot(
            subset["board_size"],
            subset["win_rate"],
            marker="o",
            label=opponent_name,
        )

    plt.ylim(0, 1)
    plt.xlabel("Board size")
    plt.ylabel("DQN win rate")
    plt.title("Phase 6A: Board-Size Transfer")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, "phase6_zero_shot_board_size_transfer.png")
    plt.savefig(path, dpi=150)
    plt.close()


def plot_starting_positions(summary_df):
    plot_df = summary_df[
        summary_df["experiment"] == "unseen_starting_positions"
    ].sort_values(["opponent", "start_stones"])

    if plot_df.empty:
        return

    plt.figure(figsize=(8, 5))

    for opponent_name in sorted(plot_df["opponent"].unique()):
        subset = plot_df[plot_df["opponent"] == opponent_name]

        plt.plot(
            subset["start_stones"],
            subset["win_rate"],
            marker="o",
            label=opponent_name,
        )

    plt.ylim(0, 1)
    plt.xlabel("Number of random pre-filled stones")
    plt.ylabel("DQN win rate")
    plt.title("Phase 6A: Unseen Starting Positions")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, "phase6_zero_shot_starting_positions.png")
    plt.savefig(path, dpi=150)
    plt.close()


def plot_random_seed_variations(summary_df):
    plot_df = summary_df[
        summary_df["experiment"] == "random_seed_opponent_variations"
    ].sort_values("opponent")

    if plot_df.empty:
        return

    plt.figure(figsize=(9, 5))
    plt.bar(plot_df["opponent"], plot_df["win_rate"])
    plt.ylim(0, 1)
    plt.xlabel("Random opponent variant")
    plt.ylabel("DQN win rate")
    plt.title("Phase 6A: Random Seed Variations")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, "phase6_zero_shot_random_seed_variations.png")
    plt.savefig(path, dpi=150)
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
        help="Which checkpoint to use if --model-path is not provided.",
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
    print("Opponents: Random Agent + Center/Greedy Agent")
    print("Result files are combined into one results CSV and one summary CSV.")
    print("=" * 80)

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

    result_frames = []

    standard_opponents = [
        ("Random Agent", lambda: RandomOpponent()),
        ("Center/Greedy Agent", lambda: CenterGreedyOpponent()),
    ]

    # ------------------------------------------------------------
    # 1. Experimental board-size transfer
    # ------------------------------------------------------------
    for board_size in args.board_sizes:
        for opponent_name, opponent_factory in standard_opponents:
            result_df = evaluate_setting(
                experiment="experimental_board_size_transfer",
                model=model,
                use_cnn=use_cnn,
                board_size=board_size,
                device=device,
                episodes=args.episodes,
                start_stones=0,
                seed=args.seed,
                opponent_name=opponent_name,
                opponent_factory=opponent_factory,
                experimental_board_transfer=True,
            )

            result_frames.append(result_df)

    # ------------------------------------------------------------
    # 2. Unseen starting positions on native 7x7
    # ------------------------------------------------------------
    for start_stones in args.start_stones:
        for opponent_name, opponent_factory in standard_opponents:
            result_df = evaluate_setting(
                experiment="unseen_starting_positions",
                model=model,
                use_cnn=use_cnn,
                board_size=TRAINED_BOARD_SIZE,
                device=device,
                episodes=args.episodes,
                start_stones=start_stones,
                seed=args.seed,
                opponent_name=opponent_name,
                opponent_factory=opponent_factory,
                experimental_board_transfer=False,
            )

            result_frames.append(result_df)

    # ------------------------------------------------------------
    # 3. Random Agent seed variations
    # ------------------------------------------------------------
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
            opponent_factory=lambda s=opponent_seed: RandomOpponent(
                seed=s,
                name=f"Random Agent seed {s}",
            ),
            experimental_board_transfer=False,
        )

        result_frames.append(result_df)

    results_df = pd.concat(result_frames, ignore_index=True)
    summary_df = summarize_results(results_df)

    results_path = os.path.join(
        RESULTS_DIR,
        "phase6_zero_shot_results.csv",
    )

    summary_path = os.path.join(
        RESULTS_DIR,
        "phase6_zero_shot_summary.csv",
    )

    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    plot_board_size_transfer(summary_df)
    plot_starting_positions(summary_df)
    plot_random_seed_variations(summary_df)

    print("\nDirect board size compatibility:")
    print(board_compatibility_df.to_string(index=False))

    print("\nCombined Phase 6 zero-shot summary:")
    print(summary_df.to_string(index=False))

    print("\nSaved files:")
    print(f"  {results_path}")
    print(f"  {summary_path}")
    print(f"  {board_compatibility_path}")
    print(f"  {os.path.join(RESULTS_DIR, 'phase6_zero_shot_board_size_compatibility.png')}")
    print(f"  {os.path.join(RESULTS_DIR, 'phase6_zero_shot_board_size_transfer.png')}")
    print(f"  {os.path.join(RESULTS_DIR, 'phase6_zero_shot_starting_positions.png')}")
    print(f"  {os.path.join(RESULTS_DIR, 'phase6_zero_shot_random_seed_variations.png')}")

    print("\nInterpretation note:")
    print(
        "This Phase 6A evaluation keeps the zero-shot task separate from the standard "
        "Phase 5 baseline evaluation. The standard 7x7 Random and Center/Greedy results "
        "remain in evaluate_phase4.py. This script evaluates zero-shot behavior across "
        "board sizes, unseen starting positions, and random seed variations. Random Agent "
        "and Center/Greedy Agent are included where meaningful. All Phase 6 result rows are "
        "stored in one combined results CSV, and all grouped metrics are stored in one "
        "combined summary CSV."
    )

    print("=" * 80)


if __name__ == "__main__":
    main()