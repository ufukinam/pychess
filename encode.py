from __future__ import annotations

"""
Encoding helpers shared by training, search, and gameplay code.

Provides:
- Board-to-tensor conversion with position history (last 4 positions).
- Fixed action space mapping (move <-> integer index).
- Legal-move masking.
- Color-flip augmentation for data augmentation during training.
"""

import numpy as np
import chess

# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------
ACTION_SIZE = 64 * 64 * 5  # from(64) * to(64) * promo(5) = 20480

PROMO_TO_IDX = {
    None: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
}
IDX_TO_PROMO = {v: k for k, v in PROMO_TO_IDX.items()}


def move_to_index(move: chess.Move) -> int:
    """Map a python-chess Move to an integer in [0, ACTION_SIZE)."""
    frm = move.from_square
    to = move.to_square
    promo = PROMO_TO_IDX.get(move.promotion, 0)
    return ((frm * 64) + to) * 5 + promo


def index_to_move(index: int) -> tuple[int, int, int]:
    """Inverse of move_to_index (returns from_sq, to_sq, promo_idx)."""
    x = index // 5
    promo_idx = index % 5
    frm = x // 64
    to = x % 64
    return frm, to, promo_idx


def action_to_move(action: int) -> chess.Move:
    """Decode integer action index back into a python-chess Move."""
    frm = action // (64 * 5)
    to = (action // 5) % 64
    promo_idx = action % 5
    promo = None if promo_idx == 0 else {
        1: chess.KNIGHT, 2: chess.BISHOP, 3: chess.ROOK, 4: chess.QUEEN,
    }[promo_idx]
    return chess.Move(frm, to, promotion=promo)


def legal_mask(board: chess.Board) -> np.ndarray:
    """Boolean vector where True marks legal actions in the global action space."""
    mask = np.zeros((ACTION_SIZE,), dtype=np.bool_)
    for mv in board.legal_moves:
        mask[move_to_index(mv)] = True
    return mask


# ---------------------------------------------------------------------------
# Board encoding with position history
# ---------------------------------------------------------------------------
HISTORY_LENGTH = 4
PIECE_PLANES = 12          # 6 piece types x 2 colors
HISTORY_PLANES = PIECE_PLANES * (1 + HISTORY_LENGTH)  # 60
META_PLANES = 7            # side(1) + castling(4) + ep(1) + halfmove(1)
IN_CHANNELS = HISTORY_PLANES + META_PLANES             # 67

_PIECE_IDX = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5,
}


def _fill_piece_planes(planes: np.ndarray, offset: int, board: chess.Board) -> None:
    """Write 12 piece-placement planes starting at *offset*."""
    for sq, piece in board.piece_map().items():
        r = 7 - chess.square_rank(sq)
        c = chess.square_file(sq)
        color_offset = 0 if piece.color == chess.WHITE else 6
        planes[offset + color_offset + _PIECE_IDX[piece.piece_type], r, c] = 1.0


def board_to_tensor(board: chess.Board, history: list[chess.Board] | None = None) -> np.ndarray:
    """
    Encode a board position (+ optional history) into a float32 tensor.

    Plane layout (IN_CHANNELS = 67):
      [ 0..11] current position pieces  (W: P N B R Q K, B: p n b r q k)
      [12..23] t-1 pieces
      [24..35] t-2 pieces
      [36..47] t-3 pieces
      [48..59] t-4 pieces
      [60]     side to move  (1 if White)
      [61]     White kingside castling
      [62]     White queenside castling
      [63]     Black kingside castling
      [64]     Black queenside castling
      [65]     en-passant file  (1s on that file column)
      [66]     halfmove clock  (normalized, clipped to [0,1])
    """
    planes = np.zeros((IN_CHANNELS, 8, 8), dtype=np.float32)

    _fill_piece_planes(planes, 0, board)

    if history:
        for i in range(min(HISTORY_LENGTH, len(history))):
            _fill_piece_planes(planes, PIECE_PLANES * (1 + i), history[-1 - i])

    base = HISTORY_PLANES
    if board.turn == chess.WHITE:
        planes[base, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.WHITE):
        planes[base + 1, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[base + 2, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[base + 3, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[base + 4, :, :] = 1.0
    if board.ep_square is not None:
        file_ = chess.square_file(board.ep_square)
        planes[base + 5, :, file_] = 1.0
    planes[base + 6, :, :] = min(board.halfmove_clock / 100.0, 1.0)

    return planes


# ---------------------------------------------------------------------------
# Color-flip augmentation  (swap White/Black + vertical mirror)
# ---------------------------------------------------------------------------

def _build_flip_action_map() -> np.ndarray:
    """Pre-compute the action-index mapping for color-flip augmentation.

    Flipping swaps ranks (sq -> sq ^ 56) while keeping files and promos.
    """
    mapping = np.zeros(ACTION_SIZE, dtype=np.int32)
    for idx in range(ACTION_SIZE):
        frm = idx // (64 * 5)
        to_sq = (idx // 5) % 64
        promo = idx % 5
        new_idx = ((frm ^ 56) * 64 + (to_sq ^ 56)) * 5 + promo
        mapping[idx] = new_idx
    return mapping


_FLIP_ACTION_MAP = _build_flip_action_map()


def augment_color_flip_state(state: np.ndarray) -> np.ndarray:
    """Return a color-flipped copy of an encoded board tensor.

    - Piece planes: swap white<->black, mirror ranks (flip axis-1).
    - Metadata: invert side-to-move, swap castling rights, keep rest.
    """
    new = np.zeros_like(state)
    flipped = state[:, ::-1, :].copy()  # mirror ranks

    for step in range(1 + HISTORY_LENGTH):
        b = PIECE_PLANES * step
        new[b:b + 6] = flipped[b + 6:b + 12]
        new[b + 6:b + 12] = flipped[b:b + 6]

    m = HISTORY_PLANES
    new[m] = 1.0 - state[m]                 # invert side-to-move
    new[m + 1] = state[m + 3]               # new WK = old BK
    new[m + 2] = state[m + 4]               # new WQ = old BQ
    new[m + 3] = state[m + 1]               # new BK = old WK
    new[m + 4] = state[m + 2]               # new BQ = old WQ
    new[m + 5] = flipped[m + 5]             # ep file (rank-flipped)
    new[m + 6] = state[m + 6]               # halfmove clock unchanged
    return new


def augment_color_flip_pi(pi: np.ndarray) -> np.ndarray:
    """Return a color-flipped copy of a policy vector."""
    new_pi = np.zeros_like(pi)
    new_pi[_FLIP_ACTION_MAP] = pi
    return new_pi
