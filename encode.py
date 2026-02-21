from __future__ import annotations

"""
Encoding helpers shared by training, search, and gameplay code.

Why this file exists:
- Neural networks work with numbers/tensors, not raw chess objects.
- MCTS also needs a fixed-size action space to store visit counts and priors.

External library usage:
- `python-chess` provides legal move generation and board metadata.
- `numpy` is used to build fast numeric arrays for model input.
"""

import numpy as np
import chess

# Fixed action space: from(64) * to(64) * promo(5) = 20480
ACTION_SIZE = 64 * 64 * 5

PROMO_TO_IDX = {
    None: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
}
IDX_TO_PROMO = {v: k for k, v in PROMO_TO_IDX.items()}


def move_to_index(move: chess.Move) -> int:
    """
    Map a `python-chess` move to one integer in `[0, ACTION_SIZE)`.

    Why: the policy head outputs one logit per action index, so moves must be
    converted into stable integer ids.
    """
    frm = move.from_square
    to = move.to_square
    promo = PROMO_TO_IDX.get(move.promotion, 0)
    return ((frm * 64) + to) * 5 + promo


def index_to_move(index: int) -> tuple[int, int, int]:
    """Inverse of `move_to_index` for debugging and action decoding."""
    x = index // 5
    promo_idx = index % 5
    frm = x // 64
    to = x % 64
    return frm, to, promo_idx


def legal_mask(board: chess.Board) -> np.ndarray:
    """
    Boolean vector over all actions where `True` marks legal actions.

    Why: the model predicts over a global action space, but only a subset is
    legal in a given position. Training and inference mask the rest.
    """
    mask = np.zeros((ACTION_SIZE,), dtype=np.bool_)
    for mv in board.legal_moves:
        mask[move_to_index(mv)] = True
    return mask


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Simple planes:
      0-5   : White P N B R Q K
      6-11  : Black p n b r q k
      12    : side-to-move (1 if white to move else 0)
      13-16 : castling rights (WK, WQ, BK, BQ)
      17    : en-passant file (one-hot across file; plane has 1s on that file)
    Output shape: `(18, 8, 8)`.

    Why these planes:
    - piece planes describe *what is on each square*.
    - side/castling/en-passant planes describe *state needed for legal play*.
    """
    planes = np.zeros((18, 8, 8), dtype=np.float32)

    piece_map = board.piece_map()
    for sq, piece in piece_map.items():
        r = 7 - chess.square_rank(sq)  # rank 8 at row 0
        c = chess.square_file(sq)
        offset = 0 if piece.color == chess.WHITE else 6
        pidx = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5,
        }[piece.piece_type]
        planes[offset + pidx, r, c] = 1.0

    # Side to move
    if board.turn == chess.WHITE:
        planes[12, :, :] = 1.0

    # Castling rights
    if board.has_kingside_castling_rights(chess.WHITE):
        planes[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[16, :, :] = 1.0

    # En-passant file
    if board.ep_square is not None:
        file_ = chess.square_file(board.ep_square)
        planes[17, :, file_] = 1.0

    return planes
