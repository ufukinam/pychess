# mcts.py  (REPLACEMENT - faster + correct tree)
from __future__ import annotations

import math
import numpy as np
import chess
import torch

from encode import ACTION_SIZE, legal_mask, board_to_tensor


def softmax_masked(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    x = logits.copy()
    x[~mask] = -1e9
    x = x - np.max(x)
    ex = np.exp(x)
    ex[~mask] = 0.0
    s = np.sum(ex)
    if s <= 0:
        p = np.zeros_like(ex, dtype=np.float32)
        legal = np.flatnonzero(mask)
        p[legal] = 1.0 / len(legal)
        return p
    return (ex / s).astype(np.float32)


def terminal_value_from_to_play(board: chess.Board) -> float:
    """
    Terminal value from perspective of player-to-move at this board.
    +1 means player-to-move is the winner in this terminal state, -1 loser, 0 draw.
    """
    res = board.result(claim_draw=True)
    if res == "1-0":
        white_value = 1.0
    elif res == "0-1":
        white_value = -1.0
    else:
        white_value = 0.0
    return white_value if board.turn == chess.WHITE else -white_value


@torch.inference_mode()
def eval_position(net, board: chess.Board, device: str = "cpu"):
    x = board_to_tensor(board)
    xt = torch.from_numpy(x).unsqueeze(0).to(device)  # (1, C, 8, 8)
    logits, value = net(xt)
    logits = logits.squeeze(0).cpu().numpy()
    value = float(value.item())

    mask = legal_mask(board)
    policy = softmax_masked(logits, mask)
    return policy, value, mask


class Node:
    __slots__ = (
        "board",
        "to_play",
        "is_expanded",
        "terminal",
        "terminal_v",
        "P",              # dict[action]->prior
        "N",              # dict[action]->visits
        "W",              # dict[action]->total value
        "children",       # dict[action]->Node
        "legal_actions",  # np.ndarray[int]
        "sum_N",
    )

    def __init__(self, board: chess.Board):
        self.board = board
        self.to_play = board.turn
        self.is_expanded = False

        self.terminal = False
        self.terminal_v = 0.0

        self.P = {}
        self.N = {}
        self.W = {}
        self.children = {}
        self.legal_actions = None
        self.sum_N = 0


def expand_node(node: Node, net, device: str, add_dirichlet: bool,
                dirichlet_alpha: float, dirichlet_eps: float) -> float:
    # Terminal?
    if node.board.is_game_over(claim_draw=True):
        node.terminal = True
        node.terminal_v = terminal_value_from_to_play(node.board)
        node.is_expanded = True
        node.legal_actions = np.array([], dtype=np.int32)
        return float(node.terminal_v)

    policy, v, mask = eval_position(net, node.board, device=device)
    legal = np.flatnonzero(mask).astype(np.int32)
    node.legal_actions = legal

    node.P = {}
    node.N = {}
    node.W = {}
    node.children = {}
    node.sum_N = 0

    for a in legal:
        a = int(a)
        node.P[a] = float(policy[a])
        node.N[a] = 0
        node.W[a] = 0.0

    if add_dirichlet and len(legal) > 0:
        noise = np.random.dirichlet([dirichlet_alpha] * len(legal))
        for i, a in enumerate(legal):
            a = int(a)
            node.P[a] = (1 - dirichlet_eps) * node.P[a] + dirichlet_eps * float(noise[i])

    node.is_expanded = True
    return float(v)



def select_action(node: Node, c_puct: float) -> int:
    # PUCT among legal_actions, using cached sum_N
    best_a = None
    best_score = -1e30

    sqrt_sum = math.sqrt(node.sum_N + 1e-8)

    for a in node.legal_actions:
        a = int(a)
        p = node.P[a]
        n = node.N[a]
        w = node.W[a]
        q = (w / n) if n > 0 else 0.0
        u = c_puct * p * (sqrt_sum / (1 + n))
        score = q + u
        if score > best_score:
            best_score = score
            best_a = a

    # legal_actions should never be empty for non-terminal
    return int(best_a)


def action_to_move(action: int) -> chess.Move:
    frm = action // (64 * 5)
    to = (action // 5) % 64
    promo_idx = action % 5
    promo = None if promo_idx == 0 else {1: chess.KNIGHT, 2: chess.BISHOP, 3: chess.ROOK, 4: chess.QUEEN}[promo_idx]
    return chess.Move(frm, to, promotion=promo)


def mcts_run(
    net,
    root: Node,
    num_sims: int = 100,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 0.3,
    dirichlet_eps: float = 0.25,
    device: str = "cpu",
):
    # Ensure root expanded (dirichlet here)
    if not root.is_expanded:
        _ = expand_node(root, net, device, add_dirichlet=True,
                        dirichlet_alpha=dirichlet_alpha, dirichlet_eps=dirichlet_eps)

    for _ in range(num_sims):
        node = root
        path = []  # list of (node, action)

        # Selection / Expansion
        while True:
            if node.terminal:
                v = node.terminal_v
                break

            if not node.is_expanded:
                v = expand_node(node, net, device, add_dirichlet=False,
                                dirichlet_alpha=dirichlet_alpha, dirichlet_eps=dirichlet_eps)
                # v is value from perspective of player-to-move at this node
                break

            # Select
            a = select_action(node, c_puct)
            path.append((node, a))

            # Descend/create child node
            child = node.children.get(a)
            if child is None:
                b2 = node.board.copy(stack=False)
                mv = action_to_move(a)
                # If mapping bug, treat as drawish
                if mv not in b2.legal_moves:
                    v = 0.0
                    break
                b2.push(mv)
                child = Node(b2)
                node.children[a] = child
            node = child

        # Backup: v from perspective of leaf player-to-move
        for parent, action in reversed(path):
            if v is None:
                v = 0.0
            parent.N[action] += 1
            parent.sum_N += 1
            # Edge statistics are stored from the parent player perspective.
            parent.W[action] += -v
            v = -v  # switch perspective each ply


def root_pi_from_visits(root: Node) -> np.ndarray:
    pi = np.zeros((ACTION_SIZE,), dtype=np.float32)
    if root.sum_N <= 0:
        return pi
    for a in root.legal_actions:
        a = int(a)
        pi[a] = root.N[a] / root.sum_N
    return pi


def pick_action_from_pi(pi: np.ndarray, temperature: float) -> int:
    if temperature <= 1e-6:
        return int(np.argmax(pi))
    p = np.power(pi, 1.0 / temperature)
    s = float(np.sum(p))
    if s <= 0:
        return int(np.argmax(pi))
    p = p / s
    return int(np.random.choice(len(p), p=p))


def mcts_policy_and_action(
    net,
    root: Node,
    num_sims: int,
    temperature: float,
    device: str = "cpu",
    c_puct: float = 1.5,
    dirichlet_alpha: float = 0.3,
    dirichlet_eps: float = 0.25,
):
    """
    Runs MCTS starting from `root` (which holds a board).
    Returns (pi, chosen_action).
    """
    mcts_run(
        net, root,
        num_sims=num_sims,
        c_puct=c_puct,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_eps=dirichlet_eps,
        device=device,
    )
    pi = root_pi_from_visits(root)
    action = pick_action_from_pi(pi, temperature)
    return pi, action


def reuse_root_after_action(root: Node, action: int) -> Node:
    """
    Tree reuse: after playing `action`, move root to the corresponding child.
    If child doesn't exist, create it.
    """
    child = root.children.get(action)
    if child is None:
        b2 = root.board.copy(stack=False)
        mv = action_to_move(action)
        if mv in b2.legal_moves:
            b2.push(mv)
        child = Node(b2)
        root.children[action] = child
    return child
