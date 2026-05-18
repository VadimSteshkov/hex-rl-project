import os
import random
import csv

import torch

from hex_engine import hexPosition, RED, BLUE, EMPTY
from models import DQN, ConvDQN, board_to_tensor, board_to_spatial_tensor
from agents import random_agent, center_agent

BOARD_SIZE = 7
N_GAMES = 200

USE_CNN = True  # <--- TOGGLE THIS TO MATCH THE MODEL YOU WANT TO EVALUATE

RESULTS_DIR = "results"
MODEL_FILE = f"phase4_model{'_cnn' if USE_CNN else ''}.pt"
CSV_FILE = f"phase4_evaluation_results{'_cnn' if USE_CNN else ''}_7x7.csv"

MODEL_PATH = os.path.join(RESULTS_DIR, MODEL_FILE)
CSV_PATH = os.path.join(RESULTS_DIR, CSV_FILE)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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

    for game_id in range(N_GAMES):
        dqn_color = RED if game_id % 2 == 0 else BLUE

        winner = play_game(
            dqn_color=dqn_color,
            opponent_agent=opponent_agent,
            model=model
        )

        if winner == dqn_color:
            dqn_wins += 1

    win_rate = dqn_wins / N_GAMES

    return {
        "opponent": opponent_name,
        "games": N_GAMES,
        "dqn_wins": dqn_wins,
        "dqn_losses": N_GAMES - dqn_wins,
        "win_rate": win_rate
    }


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Run train_phase4.py first with matching USE_CNN setting."
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Using device: {DEVICE}")
    print(f"Evaluating Model: {'ConvDQN' if USE_CNN else 'DQN'}")
    print(f"Loading model from: {MODEL_PATH}")

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

    print("\nPhase 4 Evaluation Results")
    print("-" * 60)

    for result in results:
        print(
            f"{'ConvDQN' if USE_CNN else 'DQN'} vs {result['opponent']}: "
            f"{result['dqn_wins']} / {result['games']} wins "
            f"= {result['win_rate']:.3f}"
        )

    with open(CSV_PATH, mode="w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["opponent", "games", "dqn_wins", "dqn_losses", "win_rate"]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nEvaluation results saved to: {CSV_PATH}")


if __name__ == "__main__":
    main()