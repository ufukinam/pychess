import numpy as np
import chess
import torch

from encode import move_to_index, legal_mask, ACTION_SIZE
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
        net,
        root=root,
        num_sims=1,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_eps=0.0,
        device="cpu",
    )

    visited = [int(a) for a in root.legal_actions if root.N[int(a)] > 0]
    assert len(visited) == 1, "Expected exactly one visited edge after one simulation"
    a = visited[0]
    assert np.isclose(root.W[a], -1.0), f"Expected parent edge value -1.0, got {root.W[a]:.4f}"
    print("OK: MCTS backup sign is parent-perspective correct")


if __name__ == "__main__":
    test_move_index_unique_in_startpos()
    test_mask_contains_all_legals()
    test_mcts_backup_sign()
    print("All tests passed.")
