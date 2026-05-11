import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    Simple DQN model for Hex.

    Input:
        board tensor with shape (batch_size, board_size * board_size)

    Output:
        Q-values for all board cells with shape (batch_size, board_size * board_size)
    """

    def __init__(self, board_size=5, hidden_size=128):
        super().__init__()

        self.board_size = board_size
        input_size = board_size * board_size
        output_size = board_size * board_size

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.network(x)


def board_to_tensor(board, device="cpu"):
    """
    Convert Hex board to PyTorch tensor.

    Board values:
        0  = empty
        1  = red
        -1 = blue
    """

    tensor = torch.tensor(board, dtype=torch.float32, device=device)
    tensor = tensor.flatten().unsqueeze(0)

    return tensor


def move_to_index(move, board_size):
    """
    Convert move (row, col) to scalar action index.
    """

    row, col = move
    return row * board_size + col


def index_to_move(index, board_size):
    """
    Convert scalar action index back to move (row, col).
    """

    row = index // board_size
    col = index % board_size

    return row, col