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


class ConvDQN(nn.Module):
    """
    Convolutional DQN model for Hex.

    Input:
        Spatial board tensor with shape (batch_size, 3, board_size, board_size)
        Channels: 0=RED, 1=BLUE, 2=EMPTY

    Output:
        Q-values for all board cells with shape (batch_size, board_size * board_size)
    """

    def __init__(self, board_size=5):
        super().__init__()
        self.board_size = board_size

        # Convolutional Feature Extractor
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Fully Connected Head
        flattened_size = 128 * board_size * board_size
        output_size = board_size * board_size

        self.fc = nn.Sequential(
            nn.Linear(flattened_size, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten spatial dimensions
        x = self.fc(x)
        return x


def board_to_tensor(board, device="cpu"):
    """
    Convert Hex board to flattened PyTorch tensor (For standard DQN).
    """
    tensor = torch.tensor(board, dtype=torch.float32, device=device)
    tensor = tensor.flatten().unsqueeze(0)

    return tensor


def board_to_spatial_tensor(board, device="cpu"):
    """
    Convert Hex board to a 3-channel spatial PyTorch tensor (For ConvDQN).
    Channels:
        0 = RED presence
        1 = BLUE presence
        2 = EMPTY presence
    """
    if not isinstance(board, torch.Tensor):
        board_t = torch.tensor(board, device=device)
    else:
        board_t = board.to(device)

    board_size = board_t.shape[-1]

    # Initialize zero tensor of shape [1, 3, H, W]
    tensor = torch.zeros((1, 3, board_size, board_size), dtype=torch.float32, device=device)

    # Populate channels (Assuming RED=1, BLUE=-1, EMPTY=0)
    tensor[0, 0] = (board_t == 1).float()
    tensor[0, 1] = (board_t == -1).float()
    tensor[0, 2] = (board_t == 0).float()

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