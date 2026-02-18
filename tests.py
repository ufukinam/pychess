import numpy as np
import chess

from encode import move_to_index, legal_mask, ACTION_SIZE

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

if __name__ == "__main__":
    test_move_index_unique_in_startpos()
    test_mask_contains_all_legals()
    print("All tests passed.")
