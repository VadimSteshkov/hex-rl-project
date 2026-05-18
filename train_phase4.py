import os
import random
import multiprocessing
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from hex_engine import hexPosition, RED, BLUE, EMPTY
from models import DQN, ConvDQN, board_to_spatial_tensor
from augmentation import augment_transitions_list


# =========================
# Config
# =========================

BOARD_SIZE = 7
EPISODES = 6000

GAMMA = 0.95
LR = 1e-3
BATCH_SIZE = 64
MEMORY_SIZE = 20_000

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995

TARGET_UPDATE_EVERY = 100

# Phase 5B: Data augmentation and parallelization
USE_CNN = False
USE_AUGMENTATION = True
USE_PARALLEL = True
N_WORKERS = 4
EPISODES_PER_WORKER = 10
GRAD_STEPS_PER_ROUND = 20
MIN_BUFFER = 1000
TAU = 0.005

RESULTS_DIR = "results"
MODEL_SUFFIX = "_cnn" if USE_CNN else ""
MODEL_PATH = os.path.join(RESULTS_DIR, f"phase4_model{MODEL_SUFFIX}_7x7.pt")
CURVE_PATH = os.path.join(RESULTS_DIR, f"phase4_learning_curve{MODEL_SUFFIX}.png")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

Transition = namedtuple(
    "Transition",
    ["state", "action", "reward", "next_state", "next_valid_actions", "done"]
)


# =========================
# Helper functions
# =========================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def move_to_index(move, board_size):
    row, col = move
    return row * board_size + col


def index_to_move(index, board_size):
    row = index // board_size
    col = index % board_size
    return row, col


def transform_move_for_blue(move, board_size):
    """
    Same transformation as recode_coordinates in hex_engine.py.
    Used to view BLUE as RED.
    """
    row, col = move
    return board_size - 1 - col, board_size - 1 - row


def canonical_board(board, player):
    """
    Convert board to the current agent's perspective.

    If player is RED:
        board stays the same.

    If player is BLUE:
        board is transformed so that BLUE becomes the RED-like player.
        This makes the model learn one common perspective.
    """
    size = len(board)

    if player == RED:
        return np.array(board, dtype=np.float32)

    transformed = np.zeros((size, size), dtype=np.float32)

    for row in range(size):
        for col in range(size):
            value = board[size - 1 - col][size - 1 - row]

            if value == RED:
                transformed[row, col] = BLUE
            elif value == BLUE:
                transformed[row, col] = RED
            else:
                transformed[row, col] = EMPTY

    return transformed


def canonical_action_set(action_set, player, board_size):
    """
    Convert valid moves into the canonical perspective.
    """
    if player == RED:
        return action_set

    return [transform_move_for_blue(move, board_size) for move in action_set]


def inverse_canonical_move(move, player, board_size):
    """
    Convert canonical move back to the real board coordinates.
    The transformation is symmetric, so we can reuse the same function for BLUE.
    """
    if player == RED:
        return move

    return transform_move_for_blue(move, board_size)


def state_to_tensor(state):
    """
    Convert board state to tensor.
    Shape:
        (1, board_size * board_size) if USE_CNN=False
        (1, 3, board_size, board_size) if USE_CNN=True
    """
    if USE_CNN:
        return board_to_spatial_tensor(state, device=DEVICE)
    else:
        return torch.tensor(state, dtype=torch.float32, device=DEVICE).flatten().unsqueeze(0)


def choose_action(model, state, valid_actions, epsilon):
    """
    Epsilon-greedy action selection.
    The model predicts Q-values for all cells,
    but we only select from valid actions.
    """
    board_size = state.shape[0]

    if random.random() < epsilon:
        return random.choice(valid_actions)

    with torch.no_grad():
        state_tensor = state_to_tensor(state)
        q_values = model(state_tensor).squeeze(0)

    best_move = None
    best_value = -float("inf")

    for move in valid_actions:
        action_index = move_to_index(move, board_size)
        value = q_values[action_index].item()

        if value > best_value:
            best_value = value
            best_move = move

    return best_move


def optimize_model(policy_net, target_net, optimizer, memory):
    if len(memory) < BATCH_SIZE:
        return None

    batch = random.sample(memory, BATCH_SIZE)

    states = torch.cat([
        state_to_tensor(transition.state)
        for transition in batch
    ])

    actions = torch.tensor(
        [[transition.action] for transition in batch],
        dtype=torch.long,
        device=DEVICE
    )

    rewards = torch.tensor(
        [transition.reward for transition in batch],
        dtype=torch.float32,
        device=DEVICE
    )

    dones = torch.tensor(
        [transition.done for transition in batch],
        dtype=torch.bool,
        device=DEVICE
    )

    current_q_values = policy_net(states).gather(1, actions).squeeze(1)

    next_q_values = torch.zeros(BATCH_SIZE, device=DEVICE)

    with torch.no_grad():
        for i, transition in enumerate(batch):
            if transition.done or transition.next_state is None:
                next_q_values[i] = 0.0
            else:
                next_state_tensor = state_to_tensor(transition.next_state)
                all_next_q_values = target_net(next_state_tensor).squeeze(0)

                valid_indices = [
                    move_to_index(move, BOARD_SIZE)
                    for move in transition.next_valid_actions
                ]

                next_q_values[i] = all_next_q_values[valid_indices].max()

    expected_q_values = rewards + GAMMA * next_q_values * (~dones)

    loss_fn = nn.SmoothL1Loss()
    loss = loss_fn(current_q_values, expected_q_values)

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=5.0)

    optimizer.step()

    return loss.item()


def play_training_episode(policy_net, target_net, optimizer, memory, epsilon, agent_color):
    """
    One training episode.

    The DQN agent plays either RED or BLUE.
    The opponent is random.
    """
    game = hexPosition(size=BOARD_SIZE)

    total_loss = []
    agent_won = 0

    while game.winner == EMPTY:
        if game.player == agent_color:
            current_player = game.player

            state = canonical_board(game.board, current_player)

            real_valid_actions = game.get_action_space()
            valid_actions = canonical_action_set(
                real_valid_actions,
                current_player,
                BOARD_SIZE
            )

            canonical_move = choose_action(
                policy_net,
                state,
                valid_actions,
                epsilon
            )

            real_move = inverse_canonical_move(
                canonical_move,
                current_player,
                BOARD_SIZE
            )

            action_index = move_to_index(canonical_move, BOARD_SIZE)

            game.move(real_move)

            if game.winner != EMPTY:
                reward = 1.0 if game.winner == agent_color else -1.0

                memory.append(
                    Transition(
                        state=state,
                        action=action_index,
                        reward=reward,
                        next_state=None,
                        next_valid_actions=None,
                        done=True
                    )
                )

                agent_won = 1 if game.winner == agent_color else 0
                break

            # Random opponent move
            opponent_move = random.choice(game.get_action_space())
            game.move(opponent_move)

            if game.winner != EMPTY:
                reward = -1.0

                memory.append(
                    Transition(
                        state=state,
                        action=action_index,
                        reward=reward,
                        next_state=None,
                        next_valid_actions=None,
                        done=True
                    )
                )

                agent_won = 0
                break

            next_state = canonical_board(game.board, agent_color)
            next_real_valid_actions = game.get_action_space()
            next_valid_actions = canonical_action_set(
                next_real_valid_actions,
                agent_color,
                BOARD_SIZE
            )

            memory.append(
                Transition(
                    state=state,
                    action=action_index,
                    reward=0.0,
                    next_state=next_state,
                    next_valid_actions=next_valid_actions,
                    done=False
                )
            )

            loss = optimize_model(policy_net, target_net, optimizer, memory)

            if loss is not None:
                total_loss.append(loss)

        else:
            # If the random opponent starts first
            opponent_move = random.choice(game.get_action_space())
            game.move(opponent_move)

    avg_loss = np.mean(total_loss) if total_loss else 0.0

    return agent_won, avg_loss


def polyak_update(policy_net, target_net, tau=TAU):
    """
    Soft update of target network parameters.
    target_param = (1 - tau) * target_param + tau * policy_param
    """
    with torch.no_grad():
        for target_param, policy_param in zip(
            target_net.parameters(), policy_net.parameters()
        ):
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(tau * policy_param.data)


def worker_run_episodes(args):
    """
    Worker process that generates episodes using random self-play.

    Workers play random vs. random games and collect all transitions.
    Since workers don't have access to the neural network, they use pure random play.
    The diversity of states generated is still valuable for training.

    Args:
        args: dict with keys:
            - 'n_episodes': number of episodes to generate
            - 'board_size': size of the board
            - 'seed': random seed for reproducibility

    Returns:
        List of transition dicts with keys:
        'state', 'action', 'reward', 'next_state', 'next_valid_actions', 'done'
    """
    import random as _random
    from hex_engine import hexPosition, RED, BLUE, EMPTY

    n_episodes = args['n_episodes']
    board_size = args['board_size']
    seed = args['seed']

    _random.seed(seed)
    np.random.seed(seed)

    transitions = []

    for episode in range(n_episodes):
        game = hexPosition(size=board_size)

        while game.winner == EMPTY:
            current_player = game.player
            action_space = game.get_action_space()

            state = np.array(game.board, dtype=np.float32)

            move = _random.choice(action_space)
            action_index = move[0] * board_size + move[1]

            game.move(move)

            if game.winner != EMPTY:
                reward = 1.0 if game.winner == current_player else -1.0
                transitions.append({
                    'state': state,
                    'action': action_index,
                    'reward': reward,
                    'next_state': None,
                    'next_valid_actions': None,
                    'done': True
                })
                break

            next_state = np.array(game.board, dtype=np.float32)
            next_action_space = game.get_action_space()

            transitions.append({
                'state': state,
                'action': action_index,
                'reward': 0.0,
                'next_state': next_state,
                'next_valid_actions': next_action_space,
                'done': False
            })

    return transitions


def collect_parallel_episodes(n_workers, episodes_per_worker, board_size, base_seed, pool):
    """
    Collect episodes from parallel workers.

    Args:
        n_workers: number of worker processes
        episodes_per_worker: episodes per worker
        board_size: board size
        base_seed: base seed for workers (each gets base_seed + worker_id)
        pool: multiprocessing.Pool instance

    Returns:
        List of transition dicts (empty for now since workers generate random self-play)
    """
    args_list = [
        {
            'n_episodes': episodes_per_worker,
            'board_size': board_size,
            'seed': base_seed + i,
        }
        for i in range(n_workers)
    ]

    results = pool.map(worker_run_episodes, args_list)

    all_transitions = []
    for worker_transitions in results:
        all_transitions.extend(worker_transitions)

    return all_transitions


def moving_average(values, window=100):
    if len(values) < window:
        return values

    return np.convolve(values, np.ones(window) / window, mode="valid")


# =========================
# Main training
# =========================

def main():
    set_seed(42)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Using device: {DEVICE}")
    print(f"Training DQN on Hex {BOARD_SIZE}x{BOARD_SIZE}")
    print(f"Use CNN: {USE_CNN}, Use Augmentation: {USE_AUGMENTATION}, Use Parallel: {USE_PARALLEL}")

    if USE_CNN:
        policy_net = ConvDQN(board_size=BOARD_SIZE).to(DEVICE)
        target_net = ConvDQN(board_size=BOARD_SIZE).to(DEVICE)
    else:
        policy_net = DQN(board_size=BOARD_SIZE).to(DEVICE)
        target_net = DQN(board_size=BOARD_SIZE).to(DEVICE)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = deque(maxlen=MEMORY_SIZE)

    epsilon = EPS_START
    wins = []
    losses = []

    if USE_PARALLEL:
        # Phase 5B: Parallel collection loop
        total_episodes = 0

        with multiprocessing.Pool(processes=N_WORKERS, maxtasksperchild=50) as pool:
            for round_idx in range(1, EPISODES // (N_WORKERS * EPISODES_PER_WORKER) + 1):
                transitions = collect_parallel_episodes(
                    n_workers=N_WORKERS,
                    episodes_per_worker=EPISODES_PER_WORKER,
                    board_size=BOARD_SIZE,
                    base_seed=42 + round_idx,
                    pool=pool
                )

                # Augment transitions before adding to buffer
                if USE_AUGMENTATION and transitions:
                    transitions = augment_transitions_list(transitions, BOARD_SIZE)

                # Add transitions to buffer
                for t_dict in transitions:
                    memory.append(Transition(
                        state=t_dict['state'],
                        action=t_dict['action'],
                        reward=t_dict['reward'],
                        next_state=t_dict['next_state'],
                        next_valid_actions=t_dict['next_valid_actions'],
                        done=t_dict['done']
                    ))

                # Only train after minimum buffer
                if len(memory) >= MIN_BUFFER:
                    for _ in range(GRAD_STEPS_PER_ROUND):
                        loss = optimize_model(policy_net, target_net, optimizer, memory)
                        if loss is not None:
                            losses.append(loss)

                    # Polyak update
                    polyak_update(policy_net, target_net, tau=TAU)

                epsilon = max(EPS_END, epsilon * EPS_DECAY)

                total_episodes += N_WORKERS * EPISODES_PER_WORKER

                if round_idx % 10 == 0:
                    recent_loss = np.mean(losses[-100:]) if losses else 0.0
                    print(
                        f"Round {round_idx:4d} | Episodes: {total_episodes:6d} | "
                        f"Buffer size: {len(memory):6d} | "
                        f"Loss: {recent_loss:.4f} | "
                        f"Epsilon: {epsilon:.3f}"
                    )

    else:
        # Original Phase 4: Episode-based loop
        print(f"Episodes: {EPISODES}")

        for episode in range(1, EPISODES + 1):
            agent_color = RED if episode % 2 == 0 else BLUE

            won, loss = play_training_episode(
                policy_net=policy_net,
                target_net=target_net,
                optimizer=optimizer,
                memory=memory,
                epsilon=epsilon,
                agent_color=agent_color
            )

            wins.append(won)
            losses.append(loss)

            epsilon = max(EPS_END, epsilon * EPS_DECAY)

            # Use Polyak update instead of hard copy
            polyak_update(policy_net, target_net, tau=TAU)

            if episode % 100 == 0:
                recent_win_rate = np.mean(wins[-100:])
                recent_loss = np.mean(losses[-100:])

                print(
                    f"Episode {episode:4d} | "
                    f"Win rate last 100: {recent_win_rate:.3f} | "
                    f"Loss: {recent_loss:.4f} | "
                    f"Epsilon: {epsilon:.3f}"
                )

    # Save model
    torch.save(
        {
            "model_state_dict": policy_net.state_dict(),
            "board_size": BOARD_SIZE,
            "use_cnn": USE_CNN,
            "episodes": EPISODES if not USE_PARALLEL else total_episodes,
        },
        MODEL_PATH
    )

    print(f"\nModel saved to: {MODEL_PATH}")

    # Save learning curve
    if USE_PARALLEL:
        loss_curve = moving_average(losses, window=100) if losses else losses
        plt.figure(figsize=(8, 5))
        plt.plot(loss_curve)
        plt.xlabel("Training Step")
        plt.ylabel("Loss (moving average)")
    else:
        win_rate_curve = moving_average(wins, window=100)
        plt.figure(figsize=(8, 5))
        plt.plot(win_rate_curve)
        plt.xlabel("Episode")
        plt.ylabel("Win rate moving average")

    plt.title("Phase 4 DQN Training Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(CURVE_PATH)
    plt.close()

    print(f"Learning curve saved to: {CURVE_PATH}")
    print("Training finished.")


if __name__ == "__main__":
    main()