import random
import torch

from models import board_to_tensor, index_to_move


def random_agent(board, action_set):
    """
    Baseline agent: chooses a random valid move.
    """

    return random.choice(action_set)


def center_agent(board, action_set):
    """
    Simple heuristic baseline:
    chooses the center if available,
    otherwise chooses the closest valid move to the center.
    """

    board_size = len(board)
    center = (board_size // 2, board_size // 2)

    if center in action_set:
        return center

    return min(
        action_set,
        key=lambda move: abs(move[0] - center[0]) + abs(move[1] - center[1])
    )


def dqn_agent(board, action_set, model, device="cpu"):
    """
    DQN agent:
    chooses the valid move with the highest predicted Q-value.
    """

    board_size = len(board)

    model.eval()

    with torch.no_grad():
        state = board_to_tensor(board, device=device)
        q_values = model(state).squeeze(0)

    best_move = None
    best_value = -float("inf")

    for move in action_set:
        row, col = move
        action_index = row * board_size + col
        value = q_values[action_index].item()

        if value > best_value:
            best_value = value
            best_move = move

    return best_move