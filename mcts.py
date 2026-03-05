from __future__ import annotations

"""
Monte Carlo Tree Search (MCTS) with position-history propagation.

Core loop:
1. Use the neural net to evaluate positions (policy + value).
2. Run many simulations from the current position.
3. Return root visit counts as an improved policy target.

History is threaded through the search so that each leaf evaluation
receives the full sequence of past boards (from the game + the search path).
"""

import math
import numpy as np
import chess
import torch

from encode import ACTION_SIZE, IN_CHANNELS, legal_mask, board_to_tensor, action_to_move


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def softmax_masked(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Numerically-stable softmax restricted to legal moves."""
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
    """Terminal value from the side-to-move's perspective (+1 win, -1 loss, 0 draw)."""
    res = board.result(claim_draw=True)
    if res == "1-0":
        white_value = 1.0
    elif res == "0-1":
        white_value = -1.0
    else:
        white_value = 0.0
    return white_value if board.turn == chess.WHITE else -white_value


def fast_terminal_value_from_to_play(board: chess.Board) -> float | None:
    """
    Fast terminal check from side-to-move perspective.

    Uses only automatic end conditions (checkmate/stalemate/insufficient/75-move/fivefold)
    and skips claimable-draw checks, which are significantly slower in hot MCTS paths.
    """
    if board.is_checkmate():
        return -1.0
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0
    if board.is_seventyfive_moves() or board.is_fivefold_repetition():
        return 0.0
    return None


# ------------------------------------------------------------------
# Neural-net evaluation
# ------------------------------------------------------------------

@torch.inference_mode()
def eval_position(net, board: chess.Board, device: str = "cpu",
                  history: list[chess.Board] | None = None):
    """Forward pass returning (policy, value, legal_mask)."""
    x = board_to_tensor(board, history=history)
    xt = torch.from_numpy(x).unsqueeze(0).to(device)
    logits, value = net(xt)
    logits = logits.squeeze(0).cpu().numpy()
    value = float(value.item())

    mask = legal_mask(board)
    policy = softmax_masked(logits, mask)
    return policy, value, mask


# ------------------------------------------------------------------
# Node
# ------------------------------------------------------------------

class Node:
    """One search-tree node storing edge statistics for outgoing actions."""

    __slots__ = (
        "board", "to_play", "is_expanded", "terminal", "terminal_v",
        "P", "N", "W", "children", "legal_actions", "sum_N",
    )

    def __init__(self, board: chess.Board):
        self.board = board
        self.to_play = board.turn
        self.is_expanded = False
        self.terminal = False
        self.terminal_v = 0.0
        self.P: dict[int, float] = {}
        self.N: dict[int, int] = {}
        self.W: dict[int, float] = {}
        self.children: dict[int, Node] = {}
        self.legal_actions: np.ndarray | None = None
        self.sum_N = 0


# ------------------------------------------------------------------
# Expand / Select
# ------------------------------------------------------------------

def expand_node(node: Node, net, device: str, add_dirichlet: bool,
                dirichlet_alpha: float, dirichlet_eps: float,
                history: list[chess.Board] | None = None,
                fast_terminal_checks: bool = True) -> float:
    """Evaluate and expand a leaf node, returning value from side-to-move."""
    term_v = fast_terminal_value_from_to_play(node.board) if fast_terminal_checks else None
    if term_v is None and node.board.is_game_over(claim_draw=True):
        term_v = terminal_value_from_to_play(node.board)

    if term_v is not None:
        node.terminal = True
        node.terminal_v = float(term_v)
        node.is_expanded = True
        node.legal_actions = np.array([], dtype=np.int32)
        return float(node.terminal_v)

    policy, v, mask = eval_position(net, node.board, device=device, history=history)
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

    # Allow callers (e.g., evaluation) to disable root noise by setting
    # either alpha<=0 or eps<=0.
    if add_dirichlet and len(legal) > 0 and dirichlet_alpha > 0.0 and dirichlet_eps > 0.0:
        noise = np.random.dirichlet([dirichlet_alpha] * len(legal))
        for i, a in enumerate(legal):
            a = int(a)
            node.P[a] = (1 - dirichlet_eps) * node.P[a] + dirichlet_eps * float(noise[i])

    node.is_expanded = True
    return float(v)


def select_action(node: Node, c_puct: float) -> int:
    """PUCT action selection balancing Q-value and exploration bonus."""
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

    return int(best_a)


# ------------------------------------------------------------------
# Main MCTS loop
# ------------------------------------------------------------------

def mcts_run(
    net,
    root: Node,
    num_sims: int = 100,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 0.3,
    dirichlet_eps: float = 0.25,
    device: str = "cpu",
    history: list[chess.Board] | None = None,
    fast_terminal_checks: bool = True,
):
    """
    Run *num_sims* MCTS simulations from *root*.

    *history* is the list of past board snapshots leading up to root's
    position (oldest first).  During each simulation the search path
    is appended so that leaf evaluations receive full context.
    """
    root_history = list(history) if history else []

    if not root.is_expanded:
        expand_node(root, net, device, add_dirichlet=True,
                    dirichlet_alpha=dirichlet_alpha, dirichlet_eps=dirichlet_eps,
                    history=root_history,
                    fast_terminal_checks=fast_terminal_checks)

    for _ in range(num_sims):
        node = root
        path: list[tuple[Node, int]] = []
        sim_history = list(root_history)

        while True:
            if node.terminal:
                v = node.terminal_v
                break

            if not node.is_expanded:
                v = expand_node(node, net, device, add_dirichlet=False,
                                dirichlet_alpha=dirichlet_alpha,
                                dirichlet_eps=dirichlet_eps,
                                history=sim_history,
                                fast_terminal_checks=fast_terminal_checks)
                break

            a = select_action(node, c_puct)
            path.append((node, a))
            sim_history.append(node.board)

            child = node.children.get(a)
            if child is None:
                b2 = node.board.copy(stack=False)
                mv = action_to_move(a)
                if mv not in b2.legal_moves:
                    v = 0.0
                    break
                b2.push(mv)
                child = Node(b2)
                node.children[a] = child
            node = child

        for parent, action in reversed(path):
            if v is None:
                v = 0.0
            parent.N[action] += 1
            parent.sum_N += 1
            parent.W[action] += -v
            v = -v


# ------------------------------------------------------------------
# Policy extraction & action picking
# ------------------------------------------------------------------

def root_pi_from_visits(root: Node) -> np.ndarray:
    """Convert root visit counts into a training-target policy distribution."""
    pi = np.zeros((ACTION_SIZE,), dtype=np.float32)
    if root.sum_N <= 0:
        return pi
    for a in root.legal_actions:
        a = int(a)
        pi[a] = root.N[a] / root.sum_N
    return pi


def pick_action_from_pi(pi: np.ndarray, temperature: float) -> int:
    """Sample an action from *pi* at the given temperature."""
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
    history: list[chess.Board] | None = None,
    fast_terminal_checks: bool = True,
):
    """Run MCTS then return (pi, chosen_action)."""
    mcts_run(
        net, root,
        num_sims=num_sims,
        c_puct=c_puct,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_eps=dirichlet_eps,
        device=device,
        history=history,
        fast_terminal_checks=fast_terminal_checks,
    )
    pi = root_pi_from_visits(root)
    action = pick_action_from_pi(pi, temperature)
    return pi, action


def reuse_root_after_action(root: Node, action: int) -> Node:
    """Tree reuse: advance root to the child for *action*."""
    child = root.children.get(action)
    if child is None:
        b2 = root.board.copy(stack=False)
        mv = action_to_move(action)
        if mv in b2.legal_moves:
            b2.push(mv)
        child = Node(b2)
        root.children[action] = child
    return child
