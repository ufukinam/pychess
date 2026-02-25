from __future__ import annotations

import argparse
import json
import os

import chess
import chess.pgn


def main():
    parser = argparse.ArgumentParser(
        description="Append one bad-vs-good move feedback sample from a PGN ply."
    )
    parser.add_argument("--pgn", type=str, required=True, help="PGN file path")
    parser.add_argument(
        "--ply",
        type=int,
        required=True,
        help="1-based ply index in the PGN move list (1=White's first move)",
    )
    parser.add_argument("--good_move", type=str, required=True, help="Preferred legal move in UCI")
    parser.add_argument("--out", type=str, default="feedback.jsonl", help="Output JSONL path")
    parser.add_argument(
        "--confidence",
        type=str,
        default="medium",
        choices=["low", "medium", "high"],
        help="Label confidence used by feedback loader if weight is omitted",
    )
    parser.add_argument("--weight", type=float, default=None, help="Optional explicit sample weight")
    args = parser.parse_args()

    if not os.path.exists(args.pgn):
        raise FileNotFoundError(f"PGN not found: {args.pgn}")

    with open(args.pgn, "r", encoding="utf-8") as f:
        game = chess.pgn.read_game(f)
    if game is None:
        raise ValueError(f"Could not read game from PGN: {args.pgn}")

    moves = list(game.mainline_moves())
    if args.ply < 1 or args.ply > len(moves):
        raise ValueError(f"--ply must be in [1, {len(moves)}], got {args.ply}")

    board = game.board()
    for i in range(args.ply - 1):
        board.push(moves[i])

    bad_move = moves[args.ply - 1]
    good_move = chess.Move.from_uci(args.good_move)
    if good_move not in board.legal_moves:
        raise ValueError(f"good_move '{args.good_move}' is illegal in this position")
    if good_move == bad_move:
        raise ValueError("good_move must differ from the played (bad) move")

    row = {
        "fen": board.fen(),
        "bad_move": bad_move.uci(),
        "good_move": good_move.uci(),
        "confidence": args.confidence,
    }
    if args.weight is not None:
        row["weight"] = float(args.weight)

    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

    print(
        f"Appended feedback: ply={args.ply} bad={bad_move.uci()} good={good_move.uci()} "
        f"out={args.out}"
    )


if __name__ == "__main__":
    main()
