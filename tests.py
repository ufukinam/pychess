"""
Basic unit tests for encoding, augmentation, and MCTS backup.
"""

import numpy as np
import chess
import torch

from encode import (
    move_to_index, legal_mask, action_to_move, board_to_tensor,
    ACTION_SIZE, IN_CHANNELS, HISTORY_LENGTH,
    augment_color_flip_state, augment_color_flip_pi,
)
from mcts import Node, mcts_run


def test_move_index_unique_in_startpos():
    board = chess.Board()
    idxs = [move_to_index(m) for m in board.legal_moves]
    assert len(idxs) == len(set(idxs)), "Duplicate indices for different legal moves!"
    assert all(0 <= i < ACTION_SIZE for i in idxs), "Index out of range!"
    print("OK: unique move indices in startpos")


def test_mask_contains_all_legals():
    board = chess.Board()
    mask = legal_mask(board)
    assert mask.shape == (ACTION_SIZE,)
    for m in board.legal_moves:
        assert mask[move_to_index(m)], f"Legal move missing from mask: {m}"
    print("OK: legal mask contains all legal moves")


def test_action_roundtrip():
    board = chess.Board()
    for mv in board.legal_moves:
        idx = move_to_index(mv)
        mv2 = action_to_move(idx)
        assert mv == mv2, f"Roundtrip failed: {mv} -> {idx} -> {mv2}"
    print("OK: action_to_move roundtrip")


def test_board_tensor_shape():
    board = chess.Board()
    t = board_to_tensor(board)
    assert t.shape == (IN_CHANNELS, 8, 8), f"Expected shape ({IN_CHANNELS},8,8), got {t.shape}"
    print(f"OK: board tensor shape is ({IN_CHANNELS},8,8)")


def test_board_tensor_history():
    board = chess.Board()
    history = []
    moves = list(board.legal_moves)[:4]
    for mv in moves:
        history.append(board.copy(stack=False))
        board.push(mv)
    t = board_to_tensor(board, history=history)
    assert t.shape == (IN_CHANNELS, 8, 8)
    for i in range(min(HISTORY_LENGTH, len(history))):
        hist_planes = t[12 + i * 12 : 12 + (i + 1) * 12]
        assert hist_planes.sum() > 0, f"History plane {i} is empty (should have pieces)"
    print("OK: history planes populated correctly")


def test_augment_color_flip():
    board = chess.Board()
    state = board_to_tensor(board)
    pi = np.zeros(ACTION_SIZE, dtype=np.float32)
    for mv in board.legal_moves:
        pi[move_to_index(mv)] = 1.0 / len(list(board.legal_moves))

    flipped_state = augment_color_flip_state(state)
    flipped_pi = augment_color_flip_pi(pi)

    assert flipped_state.shape == state.shape
    assert flipped_pi.shape == pi.shape
    assert np.isclose(flipped_pi.sum(), pi.sum(), atol=1e-5), "Policy mass changed after flip"
    assert not np.allclose(flipped_state, state), "Flipped state should differ from original"
    print("OK: color flip augmentation shapes and mass preserved")


class _ConstantValueNet(torch.nn.Module):
    def forward(self, x):
        b = x.shape[0]
        logits = torch.zeros((b, ACTION_SIZE), dtype=x.dtype, device=x.device)
        value = torch.ones((b,), dtype=x.dtype, device=x.device)
        return logits, value


def test_mcts_backup_sign():
    board = chess.Board()
    root = Node(board.copy(stack=False))
    net = _ConstantValueNet()

    mcts_run(
        net, root=root, num_sims=1,
        c_puct=1.5, dirichlet_alpha=0.3, dirichlet_eps=0.0,
        device="cpu",
    )

    visited = [int(a) for a in root.legal_actions if root.N[int(a)] > 0]
    assert len(visited) == 1, "Expected exactly one visited edge after one simulation"
    a = visited[0]
    assert np.isclose(root.W[a], -1.0), f"Expected parent edge value -1.0, got {root.W[a]:.4f}"
    print("OK: MCTS backup sign is parent-perspective correct")


def test_mcts_with_history():
    board = chess.Board()
    root = Node(board.copy(stack=False))
    net = _ConstantValueNet()
    history = [chess.Board()]

    mcts_run(
        net, root=root, num_sims=2,
        c_puct=1.5, dirichlet_alpha=0.3, dirichlet_eps=0.0,
        device="cpu", history=history,
    )
    total_visits = sum(root.N[int(a)] for a in root.legal_actions)
    assert total_visits == 2, f"Expected 2 total visits, got {total_visits}"
    print("OK: MCTS with history runs without error")


if __name__ == "__main__":
    test_move_index_unique_in_startpos()
    test_mask_contains_all_legals()
    test_action_roundtrip()
    test_board_tensor_shape()
    test_board_tensor_history()
    test_augment_color_flip()
    test_mcts_backup_sign()
    test_mcts_with_history()
    print("All tests passed.")
