import os
import random
import torch

from hex_engine import RED, BLUE, EMPTY
from models import DQN, board_to_tensor


_MODEL_CACHE = {}


def _get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get_model_path():
    project_root = _get_project_root()
    return os.path.join(project_root, "results", "phase4_model_7x7.pt")


def _transform_move_for_blue(move, board_size):
    row, col = move
    return board_size - 1 - col, board_size - 1 - row


def _canonical_board(board, player):
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


def _canonical_action_set(action_set, player, board_size):
    if player == RED:
        return action_set

    return [_transform_move_for_blue(move, board_size) for move in action_set]


def _inverse_canonical_move(move, player, board_size):
    if player == RED:
        return move

    return _transform_move_for_blue(move, board_size)


def _center_fallback_agent(board, action_set):
    board_size = len(board)
    center = (board_size // 2, board_size // 2)

    if center in action_set:
        return center

    return min(
        action_set,
        key=lambda move: abs(move[0] - center[0]) + abs(move[1] - center[1])
    )


def _load_model(board_size):
    if board_size in _MODEL_CACHE:
        return _MODEL_CACHE[board_size]

    model_path = _get_model_path()

    if not os.path.exists(model_path):
        return None

    checkpoint = torch.load(model_path, map_location="cpu")

    checkpoint_board_size = checkpoint.get("board_size", board_size)

    if checkpoint_board_size != board_size:
        return None

    model = DQN(board_size=board_size)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    _MODEL_CACHE[board_size] = model

    return model


def _infer_current_player(board, action_set):
    """
    The Hex engine does not pass the current player to facade.py.
    Therefore we infer it from the number of stones on the board.

    RED starts first.
    If red_count == blue_count, it is RED's turn.
    Otherwise, it is BLUE's turn.
    """
    red_count = 0
    blue_count = 0

    for row in board:
        for value in row:
            if value == RED:
                red_count += 1
            elif value == BLUE:
                blue_count += 1

    if red_count == blue_count:
        return RED

    return BLUE


def agent(board, action_set):
    """
    Final callable agent for the Hex framework.

    Required signature:
        agent(board, action_set) -> move
    """

    if not action_set:
        return None

    board_size = len(board)
    model = _load_model(board_size)

    if model is None:
        return _center_fallback_agent(board, action_set)

    player = _infer_current_player(board, action_set)

    state = _canonical_board(board, player)
    valid_actions = _canonical_action_set(action_set, player, board_size)

    with torch.no_grad():
        state_tensor = board_to_tensor(state, device="cpu")
        q_values = model(state_tensor).squeeze(0)

    best_move = None
    best_value = -float("inf")

    for move in valid_actions:
        row, col = move
        action_index = row * board_size + col
        value = q_values[action_index].item()

        if value > best_value:
            best_value = value
            best_move = move

    real_move = _inverse_canonical_move(best_move, player, board_size)

    if real_move not in action_set:
        return random.choice(action_set)

    return real_move