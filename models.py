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


class ConvDQN(nn.Module):
    """
    Convolutional DQN for Hex board game.

    Input:
        Spatial tensor with shape (batch_size, 3, board_size, board_size)
        where the 3 channels represent:
        - Channel 0: RED stone positions (1 if RED, 0 otherwise)
        - Channel 1: BLUE stone positions (1 if BLUE, 0 otherwise)
        - Channel 2: EMPTY cell positions (1 if EMPTY, 0 otherwise)

    Output:
        Q-values for all board cells with shape (batch_size, board_size * board_size)
    """

    def __init__(self, board_size=5, hidden_size=256):
        super().__init__()

        self.board_size = board_size
        self.hidden_size = hidden_size
        output_size = board_size * board_size

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        conv_output_size = 128 * board_size * board_size

        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


def board_to_spatial_tensor(board, device="cpu"):
    """
    Convert Hex board to spatial tensor for ConvDQN.

    Encodes board as a 3-channel tensor:
    - Channel 0: RED stones (1 where board==1, else 0)
    - Channel 1: BLUE stones (1 where board==-1, else 0)
    - Channel 2: EMPTY cells (1 where board==0, else 0)

    Args:
        board: 2D array/list with values {-1, 0, 1}
        device: torch device ("cpu" or "cuda")

    Returns:
        Tensor of shape (1, 3, board_size, board_size)
    """
    board_np = torch.tensor(board, dtype=torch.float32, device=device)

    board_size = board_np.shape[0]

    red_channel = (board_np == 1).float()
    blue_channel = (board_np == -1).float()
    empty_channel = (board_np == 0).float()

    spatial_tensor = torch.stack([red_channel, blue_channel, empty_channel], dim=0)
    spatial_tensor = spatial_tensor.unsqueeze(0)

    return spatial_tensor