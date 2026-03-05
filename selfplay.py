from __future__ import annotations

"""
Self-play game generation with position-history tracking.

Produces (state, improved_policy, value_target) training samples where
*state* includes the last HISTORY_LENGTH board snapshots.
"""

import os
import time
import numpy as np
import chess
import chess.pgn

from env import ChessEnv
from encode import board_to_tensor, move_to_index, action_to_move
from mcts import Node, mcts_policy_and_action, reuse_root_after_action

move_to_action = move_to_index

PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}


def material_diff_white(board: chess.Board) -> int:
    score = 0
    for pt, val in PIECE_VALUES.items():
        score += val * len(board.pieces(pt, chess.WHITE))
        score -= val * len(board.pieces(pt, chess.BLACK))
    return score


def _captured_piece_value(board: chess.Board, move: chess.Move) -> int:
    if board.is_en_passant(move):
        return PIECE_VALUES[chess.PAWN]
    p = board.piece_at(move.to_square)
    return 0 if p is None else PIECE_VALUES[p.piece_type]


def _exchange_delta_for_mover(board: chess.Board, move: chess.Move) -> int:
    if not board.is_capture(move):
        return 0
    mover = board.piece_at(move.from_square)
    attacker_val = 0 if mover is None else PIECE_VALUES[mover.piece_type]
    return _captured_piece_value(board, move) - attacker_val


def save_pgn(moves, result_str, out_dir, tags=None):
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


def _pos_key(board: chess.Board) -> str:
    return " ".join(board.fen().split(" ")[:4])


def play_self_game(
    net,
    num_sims: int = 25,
    max_plies: int = 180,
    temp_moves: int = 30,
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
    claim_draw_terminal: bool = True,
):
    EARLY_SIMS = int(early_sims) if early_sims is not None else int(num_sims)
    EARLY_PLIES = max(0, int(early_plies))
    LATE_SIMS = int(late_sims) if late_sims is not None else int(num_sims)

    env = ChessEnv()
    board = env.reset()
    root = Node(board.copy(stack=False))

    traj = []
    moves = []
    ply = 0
    board_history: list[chess.Board] = []

    captures = 0
    pawn_captures = 0
    favorable_exchanges_white = 0.0
    free_captures = 0
    threefold_claimed = 0
    ended_by_maxplies = 0
    broke_no_progress = 0
    broke_repeat2 = 0

    pos_counts: dict[str, int] = {}
    pos_counts[_pos_key(board)] = 1

    while (not board.is_game_over(claim_draw=bool(claim_draw_terminal))) and ply < max_plies:
        t = temperature if ply < temp_moves else temp_floor
        sims = EARLY_SIMS if ply < EARLY_PLIES else LATE_SIMS

        pi, action = mcts_policy_and_action(
            net, root=root, num_sims=sims, temperature=t,
            device=device, history=board_history,
        )

        state = board_to_tensor(board, history=board_history)
        traj.append((state, pi, board.turn))

        mv = action_to_move(action)
        if mv not in board.legal_moves:
            mv = np.random.choice(list(board.legal_moves))
            action = move_to_index(mv)

        if verbose:
            print(board)
            print("Move:", board.san(mv))
            print("-" * 40)

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
            b2 = board.copy(stack=False)
            b2.push(mv)
            if not b2.is_attacked_by(b2.turn, mv.to_square):
                free_captures += 1

        board_history.append(board.copy(stack=False))

        moves.append(mv)
        env.push(mv)
        board = env.board
        ply += 1

        k = _pos_key(board)
        pos_counts[k] = pos_counts.get(k, 0) + 1
        if pos_counts[k] >= 2 and stop_on_repeat2:
            broke_repeat2 = 1
            break

        if board.halfmove_clock >= int(no_progress_limit):
            broke_no_progress = 1
            break

        root = reuse_root_after_action(root, action)
        root.board = board.copy(stack=False)

        if board.is_game_over(claim_draw=bool(claim_draw_terminal)):
            root.terminal = True
            break

        if stop_on_threefold and board.can_claim_threefold_repetition():
            threefold_claimed = 1
            break

    if ply >= max_plies:
        ended_by_maxplies = 1

    # ---- outcome ----
    true_result_str = None
    if board.is_game_over(claim_draw=bool(claim_draw_terminal)):
        true_result_str = board.result(claim_draw=bool(claim_draw_terminal))
        if true_result_str == "1-0":
            z = 1.0
        elif true_result_str == "0-1":
            z = -1.0
        else:
            z = 0.0
        if z == 0.0:
            z = float(draw_penalty)
    else:
        if broke_no_progress:
            z = float(no_progress_penalty)
        elif broke_repeat2:
            z = float(repeat2_penalty)
        else:
            z = float(draw_penalty)

    if use_material_shaping:
        md = float(material_diff_white(board))
        z = float(np.clip(
            z + float(material_scale) * md + float(exchange_scale) * favorable_exchanges_white,
            -1.0, 1.0,
        ))

    if board.is_game_over(claim_draw=True) and board.can_claim_threefold_repetition():
        threefold_claimed = 1

    result_str = true_result_str if (true_result_str is not None) else "1/2-1/2"

    samples = []
    for state, pi, to_play in traj:
        v = z if to_play == chess.WHITE else -z
        samples.append((state, pi.astype(np.float32), float(v)))

    pgn_path = None
    if pgn_dir is not None:
        pgn_path = save_pgn(
            moves, result_str, pgn_dir,
            tags={
                "PlyCount": ply, "SimEarly": EARLY_SIMS,
                "EarlyPlies": EARLY_PLIES, "SimLate": LATE_SIMS,
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

    draw_like = 1 if abs(z) < 0.5 else 0
    stats = {
        "result_z_white": float(z), "result_str": result_str,
        "plies": int(ply), "pgn_path": pgn_path,
        "captures": int(captures), "pawn_captures": int(pawn_captures),
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
