"""
Data augmentation utilities for Hex board game.

Implements 180-degree board rotation as a valid augmentation strategy.
180° rotation is a true symmetry of Hex: RED's left-right connection goal
is preserved under (r,c) -> (N-1-r, N-1-c), and hexagonal adjacency is maintained.
"""

import numpy as np


def rotate_180(board_np):
    """
    Apply 180-degree rotation to a board.

    Args:
        board_np: numpy array of shape (N, N) with values in {-1, 0, 1}

    Returns:
        Rotated board of same shape.
    """
    return np.rot90(board_np, k=2).copy()


def rotate_action_180(action_index, board_size):
    """
    Map an action index under 180-degree rotation.

    Maps cell (r,c) to (N-1-r, N-1-c):
    action_index = r*N + c  ->  (N-1-r)*N + (N-1-c)

    Args:
        action_index: scalar in [0, N*N)
        board_size: N

    Returns:
        Rotated action index.
    """
    r = action_index // board_size
    c = action_index % board_size
    rotated_r = board_size - 1 - r
    rotated_c = board_size - 1 - c
    return rotated_r * board_size + rotated_c


def rotate_move_180(move, board_size):
    """
    Map a move (row, col) under 180-degree rotation.

    Args:
        move: tuple (row, col)
        board_size: N

    Returns:
        Rotated move (N-1-row, N-1-col).
    """
    row, col = move
    return (board_size - 1 - row, board_size - 1 - col)


def augment_transition(state_dict, board_size):
    """
    Create an augmented (180-rotated) version of a transition.

    Args:
        state_dict: dictionary with keys:
            - state: numpy array (N, N)
            - action: int (action index)
            - reward: float
            - next_state: numpy array (N, N) or None
            - next_valid_actions: list of (row, col) tuples or None
            - done: bool
        board_size: N (board size)

    Returns:
        Dictionary with the same structure as state_dict, with:
        - state: 180-rotated
        - action: index mapped under rotation
        - reward: unchanged (scalar)
        - next_state: 180-rotated if not None, else None
        - next_valid_actions: all moves rotated if not None, else None
        - done: unchanged
    """
    rotated = {
        'state': rotate_180(state_dict['state']),
        'action': rotate_action_180(state_dict['action'], board_size),
        'reward': state_dict['reward'],
        'next_state': rotate_180(state_dict['next_state']) if state_dict['next_state'] is not None else None,
        'next_valid_actions': [rotate_move_180(move, board_size) for move in state_dict['next_valid_actions']]
                             if state_dict['next_valid_actions'] is not None else None,
        'done': state_dict['done'],
    }
    return rotated


def augment_transitions_list(transitions, board_size):
    """
    Augment a list of transitions with 180-degree rotations.

    Args:
        transitions: list of dictionaries (or namedtuples with _asdict method)
        board_size: N

    Returns:
        List with 2x transitions: original + 180-rotated versions.
    """
    augmented = []

    for t in transitions:
        # Convert namedtuple to dict if needed
        if hasattr(t, '_asdict'):
            t_dict = t._asdict()
        else:
            t_dict = t

        # Add original
        augmented.append(t_dict)

        # Add rotated
        rotated_dict = augment_transition(t_dict, board_size)
        augmented.append(rotated_dict)

    return augmented
