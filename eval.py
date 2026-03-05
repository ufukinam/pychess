from __future__ import annotations

"""
Evaluation utilities: model vs random, model vs model.
History is tracked during eval games for accurate neural-net evaluation.
"""

import numpy as np
import chess

from env import ChessEnv
from encode import action_to_move, move_to_index, IN_CHANNELS
from mcts import Node, mcts_policy_and_action, reuse_root_after_action

EVAL_DIRICHLET_ALPHA = 0.0
EVAL_DIRICHLET_EPS = 0.0


def _apply_random_opening(
    env: ChessEnv,
    board_history: list[chess.Board],
    max_random_plies: int,
) -> int:
    """Play a random legal opening prefix and return applied plies."""
    max_p = max(0, int(max_random_plies))
    if max_p <= 0:
        return 0
    target = int(np.random.randint(0, max_p + 1))
    applied = 0
    board = env.board
    for _ in range(target):
        if env.is_terminal():
            break
        legal = list(board.legal_moves)
        if not legal:
            break
        mv = legal[int(np.random.randint(0, len(legal)))]
        board_history.append(board.copy(stack=False))
        env.push(mv)
        board = env.board
        applied += 1
        if board.is_game_over(claim_draw=True):
            break
    return applied


def play_vs_random(
    net,
    net_plays_white: bool,
    num_sims: int = 50,
    max_plies: int = 200,
    device: str = "cpu",
) -> float:
    """Returns +1 net win, 0 draw, -1 net loss."""
    env = ChessEnv()
    board = env.reset()
    root = Node(board.copy(stack=False))
    ply = 0
    board_history: list[chess.Board] = []

    while (not env.is_terminal()) and ply < max_plies:
        net_to_move = (board.turn == chess.WHITE) == net_plays_white

        if net_to_move:
            pi, action = mcts_policy_and_action(
                net, root=root, num_sims=num_sims, temperature=1e-6,
                device=device, history=board_history,
                dirichlet_alpha=EVAL_DIRICHLET_ALPHA,
                dirichlet_eps=EVAL_DIRICHLET_EPS,
            )
            mv = action_to_move(action)
            if mv not in board.legal_moves:
                mv = np.random.choice(list(board.legal_moves))
                action = move_to_index(mv)
        else:
            mv = np.random.choice(list(board.legal_moves))

        board_history.append(board.copy(stack=False))
        env.push(mv)
        board = env.board
        ply += 1

        if net_to_move:
            root = reuse_root_after_action(root, action)
            root.board = board.copy(stack=False)
        else:
            root = Node(board.copy(stack=False))

    if env.is_terminal():
        z_white = env.result_value()
    else:
        z_white = 0.0
    return float(z_white) if net_plays_white else float(-z_white)


def play_net_vs_net(
    net_a, net_b,
    a_plays_white: bool,
    num_sims: int = 50,
    max_plies: int = 200,
    device: str = "cpu",
    random_opening_plies: int = 0,
) -> float:
    """Returns +1 net_a win, 0 draw, -1 net_a loss."""
    env = ChessEnv()
    board = env.reset()
    board_history: list[chess.Board] = []
    ply = _apply_random_opening(env, board_history, int(random_opening_plies))
    board = env.board

    while (not env.is_terminal()) and ply < max_plies:
        a_to_move = (board.turn == chess.WHITE) == a_plays_white
        side_net = net_a if a_to_move else net_b

        root = Node(board.copy(stack=False))
        _, action = mcts_policy_and_action(
            side_net, root=root, num_sims=num_sims, temperature=1e-6,
            device=device, history=board_history,
            dirichlet_alpha=EVAL_DIRICHLET_ALPHA,
            dirichlet_eps=EVAL_DIRICHLET_EPS,
        )
        mv = action_to_move(action)
        if mv not in board.legal_moves:
            mv = np.random.choice(list(board.legal_moves))

        board_history.append(board.copy(stack=False))
        env.push(mv)
        board = env.board
        ply += 1

    if env.is_terminal():
        z_white = env.result_value()
    else:
        z_white = 0.0
    return float(z_white if a_plays_white else -z_white)


def eval_candidate_vs_baseline(candidate_net, baseline_net,
                               games: int = 30, num_sims: int = 50,
                               device: str = "cpu",
                               random_opening_plies: int = 0) -> dict:
    wins = draws = losses = 0
    results = []
    for i in range(games):
        r = play_net_vs_net(candidate_net, baseline_net,
                            a_plays_white=(i % 2 == 0),
                            num_sims=num_sims, device=device,
                            random_opening_plies=int(random_opening_plies))
        results.append(r)
        if r > 0: wins += 1
        elif r < 0: losses += 1
        else: draws += 1
    score = (wins + 0.5 * draws) / max(1, games)
    return {"games": games, "wins": wins, "draws": draws, "losses": losses,
            "score": float(score), "avg_result": float(np.mean(results)) if results else 0.0}


def eval_net_vs_random(net, games: int = 12, num_sims: int = 50,
                       device: str = "cpu") -> dict:
    wins = draws = losses = 0
    results = []
    for i in range(games):
        r = play_vs_random(net, net_plays_white=(i % 2 == 0),
                           num_sims=num_sims, device=device)
        results.append(r)
        if r > 0: wins += 1
        elif r < 0: losses += 1
        else: draws += 1
    score = (wins + 0.5 * draws) / max(1, games)
    return {"games": games, "wins": wins, "draws": draws, "losses": losses,
            "score": float(score), "avg_result": float(np.mean(results)) if results else 0.0}
