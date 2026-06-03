import os
import random
import csv

import torch

from hex_engine import hexPosition, RED, BLUE, EMPTY
from models import DQN, ConvDQN, board_to_tensor, board_to_spatial_tensor
from agents import random_agent, center_agent


# =========================
# Config
# =========================

BOARD_SIZE = 7
N_GAMES = 200

USE_CNN = True
USE_REWARD_SHAPING = True

RESULTS_DIR = "results"

# Phase 5E:
# False = evaluate final checkpoint
# True  = evaluate best checkpoint saved during training
LOAD_BEST_MODEL = False


# =========================
# Experiment naming
# =========================

ARCH_NAME = "cnn" if USE_CNN else "mlp"
SHAPING_NAME = "reward_shaping_soft_path_block" if USE_REWARD_SHAPING else "no_shaping"
EXPERIMENT_NAME = f"phase5_{ARCH_NAME}_{SHAPING_NAME}_{BOARD_SIZE}x{BOARD_SIZE}"

MODEL_FILE = f"{EXPERIMENT_NAME}.pt"
BEST_MODEL_FILE = f"{EXPERIMENT_NAME}_best.pt"

EVALUATION_SUFFIX = "best" if LOAD_BEST_MODEL else "final"
CSV_FILE = f"{EXPERIMENT_NAME}_{EVALUATION_SUFFIX}_evaluation_results.csv"

MODEL_PATH = os.path.join(
    RESULTS_DIR,
    BEST_MODEL_FILE if LOAD_BEST_MODEL else MODEL_FILE,
)

CSV_PATH = os.path.join(RESULTS_DIR, CSV_FILE)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Helper functions
# =========================

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


def load_model():
    if USE_CNN:
        model = ConvDQN(board_size=BOARD_SIZE).to(DEVICE)
    else:
        model = DQN(board_size=BOARD_SIZE).to(DEVICE)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def dqn_agent(board, action_set, player, model):
    state = canonical_board(board, player)
    valid_actions = canonical_action_set(action_set, player, BOARD_SIZE)

    with torch.no_grad():
        if USE_CNN:
            state_tensor = board_to_spatial_tensor(state, device=DEVICE)
        else:
            state_tensor = board_to_tensor(state, device=DEVICE)

        q_values = model(state_tensor).squeeze(0)

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


def play_game(dqn_color, opponent_agent, model):
    game = hexPosition(size=BOARD_SIZE)

    while game.winner == EMPTY:
        action_set = game.get_action_space()

        if game.player == dqn_color:
            move = dqn_agent(
                board=game.board,
                action_set=action_set,
                player=game.player,
                model=model
            )
        else:
            move = opponent_agent(game.board, action_set)

        if move not in action_set:
            move = random.choice(action_set)

        game.move(move)

    return game.winner


def evaluate_against(opponent_name, opponent_agent, model):
    dqn_wins = 0
    dqn_red_wins = 0
    dqn_blue_wins = 0

    red_games = 0
    blue_games = 0

    for game_id in range(N_GAMES):
        dqn_color = RED if game_id % 2 == 0 else BLUE

        if dqn_color == RED:
            red_games += 1
        else:
            blue_games += 1

        winner = play_game(
            dqn_color=dqn_color,
            opponent_agent=opponent_agent,
            model=model
        )

        if winner == dqn_color:
            dqn_wins += 1

            if dqn_color == RED:
                dqn_red_wins += 1
            else:
                dqn_blue_wins += 1

    win_rate = dqn_wins / N_GAMES
    red_win_rate = dqn_red_wins / red_games if red_games else 0.0
    blue_win_rate = dqn_blue_wins / blue_games if blue_games else 0.0

    return {
        "opponent": opponent_name,
        "games": N_GAMES,
        "dqn_wins": dqn_wins,
        "dqn_losses": N_GAMES - dqn_wins,
        "win_rate": win_rate,
        "red_games": red_games,
        "dqn_red_wins": dqn_red_wins,
        "red_win_rate": red_win_rate,
        "blue_games": blue_games,
        "dqn_blue_wins": dqn_blue_wins,
        "blue_win_rate": blue_win_rate,
    }


# =========================
# Main evaluation
# =========================

def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. "
            f"Run train_phase4.py first with the same experiment settings."
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Using device: {DEVICE}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Evaluating Model: {'ConvDQN' if USE_CNN else 'DQN'}")
    print(f"Reward shaping during training: {'ON' if USE_REWARD_SHAPING else 'OFF'}")
    print(f"Checkpoint type: {'best' if LOAD_BEST_MODEL else 'final'}")
    print(f"Loading model from: {MODEL_PATH}")
    print(f"CSV output: {CSV_PATH}")

    model = load_model()

    results = []

    results.append(
        evaluate_against(
            opponent_name="Random Agent",
            opponent_agent=random_agent,
            model=model
        )
    )

    results.append(
        evaluate_against(
            opponent_name="Center/Greedy Agent",
            opponent_agent=center_agent,
            model=model
        )
    )

    print("\nPhase 5 Evaluation Results")
    print("-" * 60)

    for result in results:
        print(
            f"{'ConvDQN' if USE_CNN else 'DQN'} vs {result['opponent']}: "
            f"{result['dqn_wins']} / {result['games']} wins "
            f"= {result['win_rate']:.3f} | "
            f"RED: {result['dqn_red_wins']} / {result['red_games']} "
            f"= {result['red_win_rate']:.3f} | "
            f"BLUE: {result['dqn_blue_wins']} / {result['blue_games']} "
            f"= {result['blue_win_rate']:.3f}"
        )

    with open(CSV_PATH, mode="w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "opponent",
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
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nEvaluation results saved to: {CSV_PATH}")


if __name__ == "__main__":
    main()