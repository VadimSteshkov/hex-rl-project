import os
import random
import csv
import heapq
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

USE_CNN = True  # True = CNN, False = MLP

BOARD_SIZE = 7
EPISODES = 6000

GAMMA = 0.95
LR = 1e-3
BATCH_SIZE = 64
MEMORY_SIZE = 20_000

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995

# Phase 5B: data augmentation and parallel self-play
# IMPORTANT:
# Reward shaping is only used in the episode-based training loop.
# Therefore, for the reward shaping experiment, USE_PARALLEL must be False.
USE_AUGMENTATION = True
USE_PARALLEL = False
N_WORKERS = 4
EPISODES_PER_WORKER = 10
GRAD_STEPS_PER_ROUND = 20
MIN_BUFFER = 1000
TAU = 0.005

# Reward shaping
USE_REWARD_SHAPING = True

# Softer reward shaping weights.
CENTER_REWARD_WEIGHT = 0.01
STONE_PROGRESS_REWARD = 0.005
PATH_REWARD_WEIGHT = 0.015
BLOCK_REWARD_WEIGHT = 0.01

# Clip shaped reward to keep intermediate rewards stable.
SHAPED_REWARD_CLIP = 0.05

RESULTS_DIR = "results"

# Phase 5E: training stability and monitoring
LOG_INTERVAL = 100
LR_SCHEDULER_STEP_SIZE = 1000
LR_SCHEDULER_GAMMA = 0.8

SAVE_BEST_MODEL = True
BEST_MODEL_MIN_EPISODES = 500


# =========================
# Experiment naming
# =========================

ARCH_NAME = "cnn" if USE_CNN else "mlp"
SHAPING_NAME = "reward_shaping_soft_path_block" if USE_REWARD_SHAPING else "no_shaping"

EXPERIMENT_NAME = f"phase5_{ARCH_NAME}_{SHAPING_NAME}_{BOARD_SIZE}x{BOARD_SIZE}"

MODEL_FILE = f"{EXPERIMENT_NAME}.pt"
CURVE_FILE = f"{EXPERIMENT_NAME}_learning_curve.png"
LOG_FILE = f"{EXPERIMENT_NAME}_training_log.csv"
BEST_MODEL_FILE = f"{EXPERIMENT_NAME}_best.pt"

MODEL_PATH = os.path.join(RESULTS_DIR, MODEL_FILE)
CURVE_PATH = os.path.join(RESULTS_DIR, CURVE_FILE)
LOG_PATH = os.path.join(RESULTS_DIR, LOG_FILE)
BEST_MODEL_PATH = os.path.join(RESULTS_DIR, BEST_MODEL_FILE)

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

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_to_index(move, board_size):
    row, col = move
    return row * board_size + col


def index_to_move(index, board_size):
    row = index // board_size
    col = index % board_size
    return row, col


def transform_move_for_blue(move, board_size):
    row, col = move
    return board_size - 1 - col, board_size - 1 - row


def canonical_board(board, player):
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
    if player == RED:
        return action_set

    return [transform_move_for_blue(move, board_size) for move in action_set]


def inverse_canonical_move(move, player, board_size):
    if player == RED:
        return move

    return transform_move_for_blue(move, board_size)


def state_to_tensor(state):
    if USE_CNN:
        return board_to_spatial_tensor(state, device=DEVICE)

    return torch.tensor(
        state,
        dtype=torch.float32,
        device=DEVICE
    ).flatten().unsqueeze(0)


def hex_neighbors(row, col, board_size):
    directions = [
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
    ]

    for dr, dc in directions:
        nr, nc = row + dr, col + dc

        if 0 <= nr < board_size and 0 <= nc < board_size:
            yield nr, nc


def cell_cost(value, player):
    if value == player:
        return 0

    if value == EMPTY:
        return 1

    return 1000


def shortest_path_cost(board, player, board_size):
    """
    Estimate how many empty cells are needed to connect the player's sides.

    RED connects left to right.
    BLUE connects top to bottom.
    """

    distances = np.full((board_size, board_size), np.inf)
    pq = []

    if player == RED:
        for row in range(board_size):
            cost = cell_cost(board[row][0], RED)
            distances[row][0] = cost
            heapq.heappush(pq, (cost, row, 0))

        def target_reached(row, col):
            return col == board_size - 1

    else:
        for col in range(board_size):
            cost = cell_cost(board[0][col], BLUE)
            distances[0][col] = cost
            heapq.heappush(pq, (cost, 0, col))

        def target_reached(row, col):
            return row == board_size - 1

    while pq:
        current_cost, row, col = heapq.heappop(pq)

        if current_cost > distances[row][col]:
            continue

        if target_reached(row, col):
            return current_cost

        for nr, nc in hex_neighbors(row, col, board_size):
            new_cost = current_cost + cell_cost(board[nr][nc], player)

            if new_cost < distances[nr][nc]:
                distances[nr][nc] = new_cost
                heapq.heappush(pq, (new_cost, nr, nc))

    return 1000


def compute_shaped_reward(state, state_after_agent_move, move, board_size):
    """
    Small intermediate reward used only during training.

    Final rewards stay unchanged:
        win  = +1
        loss = -1
    """

    if not USE_REWARD_SHAPING:
        return 0.0

    reward = 0.0

    row, col = move
    center = board_size // 2

    # 1. Center control reward
    distance_to_center = abs(row - center) + abs(col - center)
    max_distance = 2 * center if center > 0 else 1
    center_score = 1.0 - (distance_to_center / max_distance)
    reward += CENTER_REWARD_WEIGHT * center_score

    # 2. Stone progress reward
    own_stones_before = np.sum(state == RED)
    own_stones_after = np.sum(state_after_agent_move == RED)

    if own_stones_after > own_stones_before:
        reward += STONE_PROGRESS_REWARD

    # 3. Path potential reward
    own_path_before = shortest_path_cost(state, RED, board_size)
    own_path_after = shortest_path_cost(state_after_agent_move, RED, board_size)

    if own_path_after < own_path_before:
        path_improvement = own_path_before - own_path_after
        reward += PATH_REWARD_WEIGHT * path_improvement

    # 4. Blocking reward
    opponent_path_before = shortest_path_cost(state, BLUE, board_size)
    opponent_path_after = shortest_path_cost(state_after_agent_move, BLUE, board_size)

    if opponent_path_after > opponent_path_before:
        blocking_improvement = opponent_path_after - opponent_path_before
        reward += BLOCK_REWARD_WEIGHT * blocking_improvement

    reward = np.clip(reward, -SHAPED_REWARD_CLIP, SHAPED_REWARD_CLIP)

    return float(reward)


def choose_action(model, state, valid_actions, epsilon):
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
    game = hexPosition(size=BOARD_SIZE)

    total_loss = []
    total_shaped_reward = []
    optimizer_steps = 0
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

            state_after_agent_move = canonical_board(game.board, current_player)

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

            shaped_reward = compute_shaped_reward(
                state=state,
                state_after_agent_move=state_after_agent_move,
                move=canonical_move,
                board_size=BOARD_SIZE
            )

            total_shaped_reward.append(shaped_reward)

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
                    reward=shaped_reward,
                    next_state=next_state,
                    next_valid_actions=next_valid_actions,
                    done=False
                )
            )

            loss = optimize_model(policy_net, target_net, optimizer, memory)

            if loss is not None:
                total_loss.append(loss)
                optimizer_steps += 1

        else:
            opponent_move = random.choice(game.get_action_space())
            game.move(opponent_move)

    avg_loss = np.mean(total_loss) if total_loss else 0.0
    avg_shaped_reward = np.mean(total_shaped_reward) if total_shaped_reward else 0.0

    return agent_won, avg_loss, avg_shaped_reward, optimizer_steps


def moving_average(values, window=100):
    if len(values) < window:
        return values

    return np.convolve(values, np.ones(window) / window, mode="valid")


# =========================
# Phase 5E helpers
# =========================

def build_checkpoint(policy_net, total_episodes):
    return {
        "model_state_dict": policy_net.state_dict(),
        "board_size": BOARD_SIZE,
        "episodes": total_episodes,
        "use_cnn": USE_CNN,
        "use_reward_shaping": USE_REWARD_SHAPING,
        "use_augmentation": USE_AUGMENTATION,
        "use_parallel": USE_PARALLEL,
        "reward_components": [
            "center_control",
            "stone_progress",
            "path_potential",
            "blocking",
        ],
        "reward_weights": {
            "center": CENTER_REWARD_WEIGHT,
            "stone_progress": STONE_PROGRESS_REWARD,
            "path": PATH_REWARD_WEIGHT,
            "blocking": BLOCK_REWARD_WEIGHT,
            "clip": SHAPED_REWARD_CLIP,
        },
        "experiment_name": EXPERIMENT_NAME,
    }


def initialize_training_log(log_path):
    with open(log_path, mode="w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "phase",
                "episode",
                "round",
                "win_rate_last_100",
                "loss_last_100",
                "shaped_reward_last_100",
                "epsilon",
                "learning_rate",
                "buffer_size",
                "best_win_rate",
            ],
        )
        writer.writeheader()


def append_training_log(
    log_path,
    phase,
    episode,
    round_idx,
    win_rate_last_100,
    loss_last_100,
    shaped_reward_last_100,
    epsilon,
    learning_rate,
    buffer_size,
    best_win_rate,
):
    with open(log_path, mode="a", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "phase",
                "episode",
                "round",
                "win_rate_last_100",
                "loss_last_100",
                "shaped_reward_last_100",
                "epsilon",
                "learning_rate",
                "buffer_size",
                "best_win_rate",
            ],
        )
        writer.writerow(
            {
                "phase": phase,
                "episode": episode,
                "round": round_idx,
                "win_rate_last_100": win_rate_last_100,
                "loss_last_100": loss_last_100,
                "shaped_reward_last_100": shaped_reward_last_100,
                "epsilon": epsilon,
                "learning_rate": learning_rate,
                "buffer_size": buffer_size,
                "best_win_rate": best_win_rate,
            }
        )


# =========================
# Phase 5B: Polyak update + parallel self-play
# =========================

def polyak_update(policy_net, target_net, tau=TAU):
    with torch.no_grad():
        for target_param, policy_param in zip(
            target_net.parameters(), policy_net.parameters()
        ):
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(tau * policy_param.data)


def worker_run_episodes(args):
    import random as _random
    from hex_engine import hexPosition, EMPTY

    n_episodes = args["n_episodes"]
    board_size = args["board_size"]
    seed = args["seed"]

    _random.seed(seed)
    np.random.seed(seed)

    transitions = []

    for _episode in range(n_episodes):
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
                    "state": state,
                    "action": action_index,
                    "reward": reward,
                    "next_state": None,
                    "next_valid_actions": None,
                    "done": True,
                })
                break

            next_state = np.array(game.board, dtype=np.float32)
            next_action_space = game.get_action_space()

            transitions.append({
                "state": state,
                "action": action_index,
                "reward": 0.0,
                "next_state": next_state,
                "next_valid_actions": next_action_space,
                "done": False,
            })

    return transitions


def collect_parallel_episodes(n_workers, episodes_per_worker, board_size, base_seed, pool):
    args_list = [
        {
            "n_episodes": episodes_per_worker,
            "board_size": board_size,
            "seed": base_seed + i,
        }
        for i in range(n_workers)
    ]

    results = pool.map(worker_run_episodes, args_list)

    all_transitions = []
    for worker_transitions in results:
        all_transitions.extend(worker_transitions)

    return all_transitions


# =========================
# Main training
# =========================

def main():
    set_seed(42)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Using device: {DEVICE}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Model Architecture: {'CNN' if USE_CNN else 'MLP'}")
    print(f"Training on Hex {BOARD_SIZE}x{BOARD_SIZE}")
    print(f"Episodes: {EPISODES}")
    print(f"Reward shaping: {'ON' if USE_REWARD_SHAPING else 'OFF'}")
    print("Reward shaping components: center + stone progress + path potential + blocking")
    print(f"Shaped reward clip: +/-{SHAPED_REWARD_CLIP}")
    print(f"Augmentation: {'ON' if USE_AUGMENTATION else 'OFF'} | Parallel: {'ON' if USE_PARALLEL else 'OFF'}")
    print(f"Model output: {MODEL_PATH}")
    print(f"Best model output: {BEST_MODEL_PATH}")
    print(f"Curve output: {CURVE_PATH}")
    print(f"Training log output: {LOG_PATH}")

    if USE_REWARD_SHAPING and USE_PARALLEL:
        print(
            "WARNING: USE_PARALLEL=True means reward shaping is not applied. "
            "Set USE_PARALLEL=False for reward shaping experiments."
        )

    if USE_CNN:
        policy_net = ConvDQN(board_size=BOARD_SIZE).to(DEVICE)
        target_net = ConvDQN(board_size=BOARD_SIZE).to(DEVICE)
    else:
        policy_net = DQN(board_size=BOARD_SIZE).to(DEVICE)
        target_net = DQN(board_size=BOARD_SIZE).to(DEVICE)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=LR_SCHEDULER_STEP_SIZE,
        gamma=LR_SCHEDULER_GAMMA,
    )

    memory = deque(maxlen=MEMORY_SIZE)
    epsilon = EPS_START
    best_win_rate = -float("inf")

    initialize_training_log(LOG_PATH)

    wins = []
    losses = []
    shaped_rewards = []
    total_episodes = 0
    total_optimizer_steps = 0

    if USE_PARALLEL:
        rounds = max(1, EPISODES // (N_WORKERS * EPISODES_PER_WORKER))

        with multiprocessing.Pool(processes=N_WORKERS, maxtasksperchild=50) as pool:
            for round_idx in range(1, rounds + 1):
                transitions = collect_parallel_episodes(
                    n_workers=N_WORKERS,
                    episodes_per_worker=EPISODES_PER_WORKER,
                    board_size=BOARD_SIZE,
                    base_seed=42 + round_idx,
                    pool=pool,
                )

                if USE_AUGMENTATION and transitions:
                    transitions = augment_transitions_list(transitions, BOARD_SIZE)

                for t_dict in transitions:
                    memory.append(Transition(
                        state=t_dict["state"],
                        action=t_dict["action"],
                        reward=t_dict["reward"],
                        next_state=t_dict["next_state"],
                        next_valid_actions=t_dict["next_valid_actions"],
                        done=t_dict["done"],
                    ))

                optimizer_steps_this_round = 0

                if len(memory) >= MIN_BUFFER:
                    for _ in range(GRAD_STEPS_PER_ROUND):
                        loss = optimize_model(policy_net, target_net, optimizer, memory)

                        if loss is not None:
                            losses.append(loss)
                            optimizer_steps_this_round += 1
                            total_optimizer_steps += 1

                    polyak_update(policy_net, target_net, tau=TAU)

                epsilon = max(EPS_END, epsilon * EPS_DECAY)

                if optimizer_steps_this_round > 0:
                    scheduler.step()

                total_episodes += N_WORKERS * EPISODES_PER_WORKER

                if round_idx % 10 == 0:
                    recent_loss = np.mean(losses[-100:]) if losses else 0.0
                    current_lr = optimizer.param_groups[0]["lr"]

                    print(
                        f"Round {round_idx:4d} | Episodes: {total_episodes:6d} | "
                        f"Buffer: {len(memory):6d} | "
                        f"Loss: {recent_loss:.4f} | "
                        f"Epsilon: {epsilon:.3f} | "
                        f"LR: {current_lr:.6f} | "
                        f"Optimizer steps: {total_optimizer_steps}"
                    )

                    append_training_log(
                        log_path=LOG_PATH,
                        phase="parallel",
                        episode=total_episodes,
                        round_idx=round_idx,
                        win_rate_last_100="",
                        loss_last_100=recent_loss,
                        shaped_reward_last_100="",
                        epsilon=epsilon,
                        learning_rate=current_lr,
                        buffer_size=len(memory),
                        best_win_rate="",
                    )

    else:
        for episode in range(1, EPISODES + 1):
            agent_color = RED if episode % 2 == 0 else BLUE

            won, loss, shaped_reward, optimizer_steps = play_training_episode(
                policy_net=policy_net,
                target_net=target_net,
                optimizer=optimizer,
                memory=memory,
                epsilon=epsilon,
                agent_color=agent_color
            )

            wins.append(won)
            losses.append(loss)
            shaped_rewards.append(shaped_reward)
            total_optimizer_steps += optimizer_steps

            epsilon = max(EPS_END, epsilon * EPS_DECAY)

            if optimizer_steps > 0:
                scheduler.step()

            polyak_update(policy_net, target_net, tau=TAU)

            if episode % LOG_INTERVAL == 0:
                recent_win_rate = np.mean(wins[-100:])
                recent_loss = np.mean(losses[-100:])
                recent_shaped_reward = np.mean(shaped_rewards[-100:])
                current_lr = optimizer.param_groups[0]["lr"]

                if (
                    SAVE_BEST_MODEL
                    and episode >= BEST_MODEL_MIN_EPISODES
                    and recent_win_rate > best_win_rate
                ):
                    best_win_rate = recent_win_rate

                    torch.save(
                        build_checkpoint(policy_net, total_episodes=episode),
                        BEST_MODEL_PATH,
                    )

                    print(
                        f"New best checkpoint saved: {BEST_MODEL_PATH} "
                        f"(win rate last 100 = {best_win_rate:.3f})"
                    )

                print(
                    f"Episode {episode:4d} | "
                    f"Win rate last 100: {recent_win_rate:.3f} | "
                    f"Loss: {recent_loss:.4f} | "
                    f"Shaped reward: {recent_shaped_reward:.4f} | "
                    f"Epsilon: {epsilon:.3f} | "
                    f"LR: {current_lr:.6f} | "
                    f"Optimizer steps: {total_optimizer_steps}"
                )

                append_training_log(
                    log_path=LOG_PATH,
                    phase="episode",
                    episode=episode,
                    round_idx="",
                    win_rate_last_100=recent_win_rate,
                    loss_last_100=recent_loss,
                    shaped_reward_last_100=recent_shaped_reward,
                    epsilon=epsilon,
                    learning_rate=current_lr,
                    buffer_size=len(memory),
                    best_win_rate=best_win_rate if best_win_rate > -float("inf") else "",
                )

        total_episodes = EPISODES

    torch.save(
        build_checkpoint(policy_net, total_episodes=total_episodes),
        MODEL_PATH
    )

    print(f"\nModel saved to: {MODEL_PATH}")

    if USE_PARALLEL:
        loss_curve = moving_average(losses, window=100) if losses else losses

        plt.figure(figsize=(8, 5))
        plt.plot(loss_curve)
        plt.xlabel("Training step")
        plt.ylabel("Loss moving average")
    else:
        win_rate_curve = moving_average(wins, window=100)

        plt.figure(figsize=(8, 5))
        plt.plot(win_rate_curve)
        plt.xlabel("Episode")
        plt.ylabel("Win rate moving average")

    plt.title(f"{EXPERIMENT_NAME} Training Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(CURVE_PATH)
    plt.close()

    print(f"Learning curve saved to: {CURVE_PATH}")
    print("Training finished.")


if __name__ == "__main__":
    main()