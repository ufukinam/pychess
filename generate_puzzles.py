from __future__ import annotations

import argparse
import csv
import random

import chess


PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

EXTRA_PIECES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
EXTRA_WEIGHTS = [8, 4, 4, 3, 2]


def kings_not_touching(k1: int, k2: int) -> bool:
    return max(
        abs(chess.square_file(k1) - chess.square_file(k2)),
        abs(chess.square_rank(k1) - chess.square_rank(k2)),
    ) > 1


def random_square(excluded: set[int], rng: random.Random) -> int | None:
    free = [sq for sq in chess.SQUARES if sq not in excluded]
    if not free:
        return None
    return rng.choice(free)


def make_random_board(rng: random.Random) -> chess.Board | None:
    for _ in range(200):
        board = chess.Board(None)
        occupied: set[int] = set()

        wk = rng.choice(chess.SQUARES)
        bk = rng.choice(chess.SQUARES)
        if wk == bk or not kings_not_touching(wk, bk):
            continue
        board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
        occupied.update([wk, bk])

        for color in (chess.WHITE, chess.BLACK):
            count = rng.randint(3, 7)
            for _ in range(count):
                sq = random_square(occupied, rng)
                if sq is None:
                    break
                piece_type = rng.choices(EXTRA_PIECES, weights=EXTRA_WEIGHTS, k=1)[0]
                board.set_piece_at(sq, chess.Piece(piece_type, color))
                occupied.add(sq)

        board.turn = rng.choice([chess.WHITE, chess.BLACK])
        board.clear_stack()

        if board.is_valid() and not board.is_game_over(claim_draw=True):
            return board
    return None


def best_capture(board: chess.Board) -> tuple[chess.Move, float] | None:
    scored: list[tuple[float, chess.Move]] = []
    for mv in board.legal_moves:
        if not board.is_capture(mv):
            continue
        captured = board.piece_at(mv.to_square)
        if captured is None and board.is_en_passant(mv):
            captured = chess.Piece(chess.PAWN, not board.turn)
        if captured is None:
            continue
        attacker = board.piece_at(mv.from_square)
        if attacker is None:
            continue
        gain = PIECE_VALUES[captured.piece_type] - 0.15 * PIECE_VALUES[attacker.piece_type]
        scored.append((gain, mv))

    if not scored:
        return None
    scored.sort(key=lambda t: t[0], reverse=True)
    best_gain, best_mv = scored[0]
    second_gain = scored[1][0] if len(scored) > 1 else -999.0
    if best_gain < 2.0 or (best_gain - second_gain) < 0.7:
        return None
    return best_mv, best_gain


def generate_rows(count: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    seen: set[str] = set()

    attempts = 0
    max_attempts = count * 200
    while len(rows) < count and attempts < max_attempts:
        attempts += 1
        board = make_random_board(rng)
        if board is None:
            continue

        best = best_capture(board)
        if best is None:
            continue
        move, gain = best
        key = f"{' '.join(board.fen().split(' ')[:4])}|{move.uci()}"
        if key in seen:
            continue
        seen.add(key)

        rating = int(700 + min(2000, gain * 220))
        rows.append(
            {
                "PuzzleId": f"syn_{len(rows)+1:05d}",
                "FEN": board.fen(),
                "Moves": move.uci(),
                "Rating": str(rating),
                "Themes": "synthetic-capture",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic puzzle CSV.")
    parser.add_argument("--out", type=str, default="puzzles_synthetic.csv")
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = generate_rows(count=args.count, seed=args.seed)
    if not rows:
        raise RuntimeError("Could not generate any synthetic puzzles.")

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["PuzzleId", "FEN", "Moves", "Rating", "Themes"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} puzzles -> {args.out}")


if __name__ == "__main__":
    main()
