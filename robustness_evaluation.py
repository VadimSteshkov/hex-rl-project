"""
Robustness evaluation of the trained Hex agent under three perturbations:
input noise, Q-value noise, and a suboptimal opponent (epsilon-random).

This script uses the same model path and canonical RED/BLUE logic as
evaluate_phase4.py. Therefore, make sure evaluate_phase4.py points to the
mixed-opponent model before running this script.

Outputs:
    results/<experiment_name>_robustness_results.csv
    results/<experiment_name>_robustness_input_noise.png
    results/<experiment_name>_robustness_q_noise.png
    results/<experiment_name>_robustness_opponent_suboptimality.png
"""

import os
import csv
import random

import torch

from hex_engine import hexPosition, RED, EMPTY
from models import board_to_tensor, board_to_spatial_tensor
from agents import random_agent, center_agent

# Reuse config + helpers from the standard evaluation so the policy is identical.
from evaluate_phase4 import (
    BOARD_SIZE,
    USE_CNN,
    DEVICE,
    RESULTS_DIR,
    EXPERIMENT_NAME,
    MODEL_PATH,
    canonical_board,
    canonical_action_set,
    inverse_canonical_move,
    load_model,
)


# =========================
# Config
# =========================

N_GAMES = 100
SEED = 42

INPUT_NOISE_LEVELS = [0.0, 0.05, 0.1, 0.2, 0.5]
Q_NOISE_LEVELS = [0.0, 0.05, 0.1, 0.2, 0.5]
OPPONENT_EPSILONS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]

CSV_FILE = f"{EXPERIMENT_NAME}_robustness_results.csv"
CSV_PATH = os.path.join(RESULTS_DIR, CSV_FILE)


# =========================
# Reproducibility
# =========================

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================
# Perturbed agents
# =========================

def make_noisy_dqn_agent(model, input_sigma=0.0, q_sigma=0.0):
    """
    Build a DQN agent closure that injects Gaussian noise into the input
    observation and/or the predicted Q-values before choosing the greedy move.

    Signature:
        agent(board, action_set, player) -> move
    """

    def agent(board, action_set, player):
        state = canonical_board(board, player)
        valid_actions = canonical_action_set(action_set, player, BOARD_SIZE)

        with torch.no_grad():
            if USE_CNN:
                state_tensor = board_to_spatial_tensor(state, device=DEVICE)
            else:
                state_tensor = board_to_tensor(state, device=DEVICE)

            if input_sigma > 0.0:
                state_tensor = state_tensor + torch.randn_like(state_tensor) * input_sigma

            q_values = model(state_tensor).squeeze(0)

            if q_sigma > 0.0:
                q_values = q_values + torch.randn_like(q_values) * q_sigma

        best_move = None
        best_value = -float("inf")

        for move in valid_actions:
            row, col = move
            action_index = row * BOARD_SIZE + col
            value = q_values[action_index].item()

            if value > best_value:
                best_value = value
                best_move = move

        real_move = inverse_canonical_move(best_move, player, BOARD_SIZE)

        if real_move not in action_set:
            return random.choice(action_set)

        return real_move

    return agent


def make_epsilon_opponent(base_agent, epsilon):
    """
    Wrap a base opponent so that with probability epsilon it plays a uniform
    random valid move instead of its normal choice.

    epsilon = 0.0 -> pure base agent
    epsilon = 1.0 -> fully random opponent

    Signature:
        agent(board, action_set) -> move
    """

    def agent(board, action_set):
        if epsilon > 0.0 and random.random() < epsilon:
            return random.choice(action_set)

        return base_agent(board, action_set)

    return agent


# =========================
# Game simulation
# =========================

def play_game(dqn_color, dqn_agent_fn, opponent_agent):
    game = hexPosition(size=BOARD_SIZE)

    while game.winner == EMPTY:
        action_set = game.get_action_space()

        if game.player == dqn_color:
            move = dqn_agent_fn(game.board, action_set, game.player)
        else:
            move = opponent_agent(game.board, action_set)

        if move not in action_set:
            move = random.choice(action_set)

        game.move(move)

    return game.winner


def evaluate(dqn_agent_fn, opponent_agent, n_games=N_GAMES):
    dqn_wins = 0
    dqn_red_wins = 0
    dqn_blue_wins = 0

    red_games = 0
    blue_games = 0

    for game_id in range(n_games):
        dqn_color = RED if game_id % 2 == 0 else -RED

        if dqn_color == RED:
            red_games += 1
        else:
            blue_games += 1

        winner = play_game(
            dqn_color=dqn_color,
            dqn_agent_fn=dqn_agent_fn,
            opponent_agent=opponent_agent,
        )

        if winner == dqn_color:
            dqn_wins += 1

            if dqn_color == RED:
                dqn_red_wins += 1
            else:
                dqn_blue_wins += 1

    return {
        "games": n_games,
        "dqn_wins": dqn_wins,
        "win_rate": dqn_wins / n_games,
        "red_win_rate": dqn_red_wins / red_games if red_games else 0.0,
        "blue_win_rate": dqn_blue_wins / blue_games if blue_games else 0.0,
    }


# =========================
# Sweeps
# =========================

OPPONENTS = [
    ("Random Agent", random_agent),
    ("Center/Greedy Agent", center_agent),
]


def run_input_noise_sweep(model, rows):
    print("\n[1/3] Input noise sweep")
    print("-" * 60)

    for sigma in INPUT_NOISE_LEVELS:
        dqn_agent_fn = make_noisy_dqn_agent(
            model=model,
            input_sigma=sigma,
            q_sigma=0.0,
        )

        for opponent_name, opponent_agent in OPPONENTS:
            result = evaluate(
                dqn_agent_fn=dqn_agent_fn,
                opponent_agent=opponent_agent,
            )

            print(
                f"input_sigma={sigma:<5} vs {opponent_name:<20} "
                f"win_rate={result['win_rate']:.3f} "
                f"(R={result['red_win_rate']:.3f} B={result['blue_win_rate']:.3f})"
            )

            rows.append({
                "test_type": "input_noise",
                "param_name": "input_sigma",
                "param_value": sigma,
                "opponent": opponent_name,
                "games": result["games"],
                "dqn_wins": result["dqn_wins"],
                "win_rate": result["win_rate"],
                "red_win_rate": result["red_win_rate"],
                "blue_win_rate": result["blue_win_rate"],
            })


def run_q_noise_sweep(model, rows):
    print("\n[2/3] Q-signal noise sweep")
    print("-" * 60)

    for sigma in Q_NOISE_LEVELS:
        dqn_agent_fn = make_noisy_dqn_agent(
            model=model,
            input_sigma=0.0,
            q_sigma=sigma,
        )

        for opponent_name, opponent_agent in OPPONENTS:
            result = evaluate(
                dqn_agent_fn=dqn_agent_fn,
                opponent_agent=opponent_agent,
            )

            print(
                f"q_sigma={sigma:<5} vs {opponent_name:<20} "
                f"win_rate={result['win_rate']:.3f} "
                f"(R={result['red_win_rate']:.3f} B={result['blue_win_rate']:.3f})"
            )

            rows.append({
                "test_type": "q_noise",
                "param_name": "q_sigma",
                "param_value": sigma,
                "opponent": opponent_name,
                "games": result["games"],
                "dqn_wins": result["dqn_wins"],
                "win_rate": result["win_rate"],
                "red_win_rate": result["red_win_rate"],
                "blue_win_rate": result["blue_win_rate"],
            })


def run_opponent_suboptimality_sweep(model, rows):
    print("\n[3/3] Opponent suboptimality sweep")
    print("-" * 60)

    dqn_agent_fn = make_noisy_dqn_agent(model=model)

    opponent_name = "Center/Greedy (epsilon-random)"

    for epsilon in OPPONENT_EPSILONS:
        opponent_agent = make_epsilon_opponent(
            base_agent=center_agent,
            epsilon=epsilon,
        )

        result = evaluate(
            dqn_agent_fn=dqn_agent_fn,
            opponent_agent=opponent_agent,
        )

        print(
            f"opponent_eps={epsilon:<5} "
            f"win_rate={result['win_rate']:.3f} "
            f"(R={result['red_win_rate']:.3f} B={result['blue_win_rate']:.3f})"
        )

        rows.append({
            "test_type": "opponent_suboptimality",
            "param_name": "opponent_epsilon",
            "param_value": epsilon,
            "opponent": opponent_name,
            "games": result["games"],
            "dqn_wins": result["dqn_wins"],
            "win_rate": result["win_rate"],
            "red_win_rate": result["red_win_rate"],
            "blue_win_rate": result["blue_win_rate"],
        })


# =========================
# Plotting
# =========================

def plot_sweep(rows, test_type, param_name, title, filename):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plot '{filename}' because matplotlib is unavailable: {exc}")
        return

    subset = [row for row in rows if row["test_type"] == test_type]
    opponents = sorted({row["opponent"] for row in subset})

    plt.figure(figsize=(7, 5))

    for opponent_name in opponents:
        opponent_rows = sorted(
            [row for row in subset if row["opponent"] == opponent_name],
            key=lambda row: row["param_value"],
        )

        x_values = [row["param_value"] for row in opponent_rows]
        y_values = [row["win_rate"] for row in opponent_rows]

        plt.plot(
            x_values,
            y_values,
            marker="o",
            label=opponent_name,
        )

    plt.xlabel(param_name)
    plt.ylabel("DQN win rate")
    plt.title(title)
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=120)
    plt.close()

    print(f"Saved plot: {path}")


# =========================
# Main
# =========================

def main():
    set_seed(SEED)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. "
            f"Make sure evaluate_phase4.py points to the mixed-opponent model."
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Using device: {DEVICE}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Model: {'ConvDQN' if USE_CNN else 'DQN'}")
    print(f"Loading model from: {MODEL_PATH}")
    print(f"Games per setting: {N_GAMES}")
    print(f"CSV output: {CSV_PATH}")

    model = load_model()

    rows = []

    run_input_noise_sweep(model, rows)
    run_q_noise_sweep(model, rows)
    run_opponent_suboptimality_sweep(model, rows)

    with open(CSV_PATH, mode="w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "test_type",
                "param_name",
                "param_value",
                "opponent",
                "games",
                "dqn_wins",
                "win_rate",
                "red_win_rate",
                "blue_win_rate",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"\nRobustness results saved to: {CSV_PATH}")

    plot_sweep(
        rows=rows,
        test_type="input_noise",
        param_name="input_sigma",
        title="Robustness to input noise",
        filename=f"{EXPERIMENT_NAME}_robustness_input_noise.png",
    )

    plot_sweep(
        rows=rows,
        test_type="q_noise",
        param_name="q_sigma",
        title="Robustness to Q-signal noise",
        filename=f"{EXPERIMENT_NAME}_robustness_q_noise.png",
    )

    plot_sweep(
        rows=rows,
        test_type="opponent_suboptimality",
        param_name="opponent_epsilon",
        title="Reaction to sub-optimal / random opponents",
        filename=f"{EXPERIMENT_NAME}_robustness_opponent_suboptimality.png",
    )

    print("Robustness evaluation finished.")


if __name__ == "__main__":
    main()