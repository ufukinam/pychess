from __future__ import annotations

import argparse
import glob
import json
import os

import chess
import chess.pgn


def iter_pgn_paths(pattern: str, recursive: bool) -> list[str]:
    return sorted(glob.glob(pattern, recursive=recursive))


def read_games_from_pgn(path: str):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            yield game


def main():
    parser = argparse.ArgumentParser(
        description="Batch-generate candidate bad-move rows from many PGNs for faster labeling."
    )
    parser.add_argument(
        "--pgn_glob",
        type=str,
        default="model_games/*.pgn",
        help="Glob pattern for input PGNs (use quotes for ** patterns).",
    )
    parser.add_argument("--recursive", action="store_true", help="Enable recursive glob matching.")
    parser.add_argument("--out", type=str, default="feedback_candidates.jsonl", help="Output JSONL path.")
    parser.add_argument("--max_games", type=int, default=0, help="Cap number of games processed (0 = no cap).")
    parser.add_argument(
        "--max_plies_per_game",
        type=int,
        default=0,
        help="Cap candidate plies per game (0 = all plies).",
    )
    parser.add_argument(
        "--min_ply",
        type=int,
        default=1,
        help="Only include plies >= min_ply (1-based).",
    )
    parser.add_argument(
        "--side",
        type=str,
        default="both",
        choices=["both", "white", "black"],
        help="Only include candidate plies from selected side.",
    )
    parser.add_argument(
        "--max_legal_moves",
        type=int,
        default=20,
        help="How many legal move UCI strings to include as hint (0 disables).",
    )
    args = parser.parse_args()

    paths = iter_pgn_paths(args.pgn_glob, recursive=bool(args.recursive))
    if not paths:
        raise FileNotFoundError(f"No PGNs matched pattern: {args.pgn_glob}")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    games_seen = 0
    rows_written = 0
    max_games = int(args.max_games)
    max_plies_per_game = int(args.max_plies_per_game)
    min_ply = max(1, int(args.min_ply))
    max_legal = max(0, int(args.max_legal_moves))

    with open(args.out, "w", encoding="utf-8") as out_f:
        for pgn_path in paths:
            for game in read_games_from_pgn(pgn_path):
                if max_games > 0 and games_seen >= max_games:
                    break

                games_seen += 1
                board = game.board()
                plies_in_game = 0
                moves = list(game.mainline_moves())
                headers = dict(game.headers)

                for ply_idx, move in enumerate(moves, start=1):
                    if ply_idx < min_ply:
                        board.push(move)
                        continue

                    mover_is_white = board.turn == chess.WHITE
                    if args.side == "white" and not mover_is_white:
                        board.push(move)
                        continue
                    if args.side == "black" and mover_is_white:
                        board.push(move)
                        continue

                    san = board.san(move)
                    legal_moves = sorted(m.uci() for m in board.legal_moves)
                    if max_legal > 0:
                        legal_moves = legal_moves[:max_legal]
                    else:
                        legal_moves = []

                    row = {
                        "fen": board.fen(),
                        "bad_move": move.uci(),
                        "good_move": "",
                        "confidence": "medium",
                        "source_pgn": pgn_path,
                        "game_index": games_seen,
                        "ply": ply_idx,
                        "san": san,
                        "turn": "white" if mover_is_white else "black",
                        "result": headers.get("Result", "*"),
                        "white": headers.get("White", ""),
                        "black": headers.get("Black", ""),
                    }
                    if legal_moves:
                        row["legal_moves_uci"] = legal_moves

                    out_f.write(json.dumps(row) + "\n")
                    rows_written += 1
                    plies_in_game += 1

                    board.push(move)

                    if max_plies_per_game > 0 and plies_in_game >= max_plies_per_game:
                        break

            if max_games > 0 and games_seen >= max_games:
                break

    print(
        f"Generated {rows_written} candidate rows from {games_seen} games into {args.out}"
    )


if __name__ == "__main__":
    main()
