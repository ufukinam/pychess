import chess
import numpy as np
from encode import move_to_index, legal_mask, ACTION_SIZE

def test_pawn_capture_encoded():
    # Position where white pawn can capture: e4 pawn can capture d5
    board = chess.Board()
    board.push_san("e4")
    board.push_san("d5")  # now white pawn on e4 can capture d5

    mv = chess.Move.from_uci("e4d5")
    assert mv in board.legal_moves, "Pawn capture should be legal in this position."

    idx = move_to_index(mv)
    assert 0 <= idx < ACTION_SIZE

    mask = legal_mask(board)
    assert mask[idx], "Pawn capture move index not marked legal. Encoding/mask bug."

    print("OK: pawn capture is legal and correctly encoded/masked")

if __name__ == "__main__":
    test_pawn_capture_encoded()
