from __future__ import annotations

"""
Self-play game generation for reinforcement learning data.

Why this module matters:
- Produces `(state, improved_policy, value_target)` training samples.
- Encodes extra training-time anti-loop and shaping heuristics to reduce
  degenerate draw behavior in early training.
"""

import os
import time
import numpy as np
import chess
import chess.pgn

from env import ChessEnv
from encode import board_to_tensor
from mcts import Node, mcts_policy_and_action, reuse_root_after_action


UNICODE_PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def material_diff_white(board: chess.Board) -> int:
    """(White material) - (Black material) using simple piece values."""
    score = 0
    for piece_type, val in UNICODE_PIECE_VALUES.items():
        score += val * len(board.pieces(piece_type, chess.WHITE))
        score -= val * len(board.pieces(piece_type, chess.BLACK))
    return score


def move_to_action(move: chess.Move) -> int:
    promo_idx = {
        None: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
    }[move.promotion]
    return move.from_square * (64 * 5) + move.to_square * 5 + promo_idx


def _captured_piece_value(board: chess.Board, move: chess.Move) -> int:
    if board.is_en_passant(move):
        return UNICODE_PIECE_VALUES[chess.PAWN]
    p = board.piece_at(move.to_square)
    if p is None:
        return 0
    return UNICODE_PIECE_VALUES[p.piece_type]


def _exchange_delta_for_mover(board: chess.Board, move: chess.Move) -> int:
    """Simple exchange proxy: captured-value minus attacker-value."""
    if not board.is_capture(move):
        return 0
    mover_piece = board.piece_at(move.from_square)
    attacker_val = 0 if mover_piece is None else UNICODE_PIECE_VALUES[mover_piece.piece_type]
    captured_val = _captured_piece_value(board, move)
    return captured_val - attacker_val


def save_pgn(moves: list[chess.Move], result_str: str, out_dir: str, tags: dict | None = None) -> str:
    """Save one self-play game as PGN with optional debug/training tags."""
    os.makedirs(out_dir, exist_ok=True)

    game = chess.pgn.Game()
    game.headers["Event"] = "SelfPlay"
    game.headers["Site"] = "Local"
    game.headers["Date"] = time.strftime("%Y.%m.%d")
    game.headers["Round"] = "-"
    game.headers["White"] = "Net"
    game.headers["Black"] = "Net"
    game.headers["Result"] = result_str

    if tags:
        for k, v in tags.items():
            game.headers[str(k)] = str(v)

    node = game
    board = chess.Board()
    for mv in moves:
        node = node.add_variation(mv)
        board.push(mv)

    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"game_{ts}_{np.random.randint(100000):05d}.pgn")
    with open(path, "w", encoding="utf-8") as f:
        print(game, file=f, end="\n\n")
    return path


def action_to_move(action: int) -> chess.Move:
    frm = action // (64 * 5)
    to = (action // 5) % 64
    promo_idx = action % 5
    promo = None if promo_idx == 0 else {
        1: chess.KNIGHT,
        2: chess.BISHOP,
        3: chess.ROOK,
        4: chess.QUEEN,
    }[promo_idx]
    return chess.Move(frm, to, promotion=promo)


def _pos_key(board: chess.Board) -> str:
    """
    Key for repetition based on the parts relevant to repetition:
    piece placement, side to move, castling rights, en-passant square.
    (Ignore halfmove/fullmove counters.)
    """
    return " ".join(board.fen().split(" ")[:4])


def play_self_game(
    net,
    num_sims: int = 25,
    max_plies: int = 180,
    temp_moves: int = 12,
    temperature: float = 1.0,
    device: str = "cpu",
    pgn_dir: str | None = "games",
    verbose: bool = False,
    draw_penalty: float = 0.0,
    stop_on_threefold: bool = True,
    no_progress_limit: int = 30,
    no_progress_penalty: float = 0.0,
    repeat2_penalty: float = 0.0,
    stop_on_repeat2: bool = False,
    temp_floor: float = 0.10,
    use_material_shaping: bool = False,
    material_scale: float = 0.0,
    exchange_scale: float = 0.0,
    early_sims: int | None = None,
    early_plies: int = 16,
    late_sims: int | None = None,
):
    """
    Play one net-vs-net game and return training samples + stats.

    Returns:
    - samples: list of `(state, pi, value)` tuples for replay
    - stats: game metadata useful for monitoring behavior
    - pgn_path: saved PGN file path if enabled
    """
    # ---- MCTS schedule ----
    EARLY_SIMS = int(early_sims) if early_sims is not None else int(num_sims)
    EARLY_PLIES = max(0, int(early_plies))
    LATE_SIMS = int(late_sims) if late_sims is not None else int(num_sims)

    env = ChessEnv()
    board = env.reset()

    root = Node(board.copy(stack=False))

    traj = []
    moves = []
    ply = 0

    captures = 0
    pawn_captures = 0
    favorable_exchanges_white = 0.0
    free_captures = 0

    threefold_claimed = 0
    ended_by_maxplies = 0
    broke_no_progress = 0
    broke_repeat2 = 0

    # track repeats earlier than 3-fold
    pos_counts: dict[str, int] = {}
    pos_counts[_pos_key(board)] = 1

    while (not env.is_terminal()) and ply < max_plies:
        t = temperature if ply < temp_moves else temp_floor
        sims = EARLY_SIMS if ply < EARLY_PLIES else LATE_SIMS

        pi, action = mcts_policy_and_action(
            net,
            root=root,
            num_sims=sims,
            temperature=t,
            device=device,
        )

        state = board_to_tensor(board)
        traj.append((state, pi, board.turn))

        mv = action_to_move(action)
        if mv not in board.legal_moves:
            mv = np.random.choice(list(board.legal_moves))
            action = move_to_action(mv)

        if verbose:
            print(board)
            print("Move:", board.san(mv))
            print("-" * 40)

        # capture stats BEFORE push
        if board.is_capture(mv):
            captures += 1
            p = board.piece_at(mv.from_square)
            if p is not None and p.piece_type == chess.PAWN:
                pawn_captures += 1

            ex_delta = _exchange_delta_for_mover(board, mv)
            if board.turn == chess.WHITE:
                favorable_exchanges_white += float(ex_delta)
            else:
                favorable_exchanges_white -= float(ex_delta)

            # "Free capture" proxy: destination not currently attacked after capture.
            b2 = board.copy(stack=False)
            b2.push(mv)
            if not b2.is_attacked_by(b2.turn, mv.to_square):
                free_captures += 1

        # play move
        moves.append(mv)
        env.push(mv)
        board = env.board
        ply += 1

        # update repeat table AFTER move
        k = _pos_key(board)
        pos_counts[k] = pos_counts.get(k, 0) + 1
        if pos_counts[k] >= 2 and stop_on_repeat2:
            broke_repeat2 = 1
            break

        # no-progress stop AFTER move
        if board.halfmove_clock >= int(no_progress_limit):
            broke_no_progress = 1
            break

        # tree reuse
        root = reuse_root_after_action(root, action)
        root.board = board.copy(stack=False)

        # terminal?
        if board.is_game_over(claim_draw=True):
            root.terminal = True
            break

        # threefold stop AFTER move
        if stop_on_threefold and board.can_claim_threefold_repetition():
            threefold_claimed = 1
            break

    if ply >= max_plies:
        ended_by_maxplies = 1

    # ---- Outcome target (White perspective) ----
    true_result_str = None
    if env.is_terminal():
        true_result_str = board.result(claim_draw=True)
        z = float(env.result_value())
        if z == 0.0:
            z = float(draw_penalty)
    else:
        # stopped early by our rules or max plies
        if broke_no_progress:
            z = float(no_progress_penalty)
        elif broke_repeat2:
            z = float(repeat2_penalty)
        else:
            z = float(draw_penalty)

    # ---- Material shaping ----
    if use_material_shaping:
        md = float(material_diff_white(board))
        z = float(np.clip(
            z + float(material_scale) * md + float(exchange_scale) * favorable_exchanges_white,
            -1.0,
            1.0,
        ))

    # Keep threefold stats accurate even when game ended via generic
    # is_game_over(claim_draw=True) branch above.
    if env.is_terminal() and board.can_claim_threefold_repetition():
        threefold_claimed = 1

    # PGN result string:
    # - Use true game result for terminal positions.
    # - Force draw for non-terminal stops (max plies / loop-control rules),
    #   even if shaped z is slightly biased for training.
    if env.is_terminal():
        result_str = true_result_str if true_result_str is not None else "1/2-1/2"
    else:
        result_str = "1/2-1/2"

    samples = []
    for state, pi, to_play in traj:
        v = z if to_play == chess.WHITE else -z
        samples.append((state, pi.astype(np.float32), float(v)))

    pgn_path = None
    if pgn_dir is not None:
        pgn_path = save_pgn(
            moves,
            result_str,
            pgn_dir,
            tags={
                "PlyCount": ply,
                "SimEarly": EARLY_SIMS,
                "EarlyPlies": EARLY_PLIES,
                "SimLate": LATE_SIMS,
                "DrawPenalty": float(draw_penalty),
                "NoProgLimit": int(no_progress_limit),
                "NoProgPen": float(no_progress_penalty),
                "Repeat2Pen": float(repeat2_penalty),
                "StopOnRepeat2": int(stop_on_repeat2),
                "TempFloor": float(temp_floor),
                "MatScale": float(material_scale) if use_material_shaping else 0.0,
                "ExchScale": float(exchange_scale) if use_material_shaping else 0.0,
                "HalfmoveClockEnd": int(board.halfmove_clock),
                "BrokeNoProg": broke_no_progress,
                "BrokeRepeat2": broke_repeat2,
                "Threefold": threefold_claimed,
                "FavExWhite": round(float(favorable_exchanges_white), 3),
                "FreeCaps": int(free_captures),
            },
        )

    draw_like = 1 if (abs(z) < 0.5) else 0

    stats = {
        "result_z_white": float(z),
        "result_str": result_str,
        "plies": int(ply),
        "pgn_path": pgn_path,
        "captures": int(captures),
        "pawn_captures": int(pawn_captures),
        "free_captures": int(free_captures),
        "favorable_exchanges_white": float(favorable_exchanges_white),
        "threefold_claimed": int(threefold_claimed),
        "draw_like": int(draw_like),
        "ended_by_maxplies": int(ended_by_maxplies),
        "broke_no_progress": int(broke_no_progress),
        "broke_repeat2": int(broke_repeat2),
        "halfmove_clock_end": int(board.halfmove_clock),
    }
    return samples, stats, pgn_path
