from __future__ import annotations

"""
Puzzle CSV parsing and train/validation split logic.

External library usage:
- `csv`: reads puzzle rows from disk.
- `python-chess`: validates FEN positions and UCI moves.
- `numpy`: stores encoded state/mask arrays in examples.
"""

from dataclasses import dataclass
from typing import Iterable
import csv
import random

import chess
import numpy as np

from encode import ACTION_SIZE, board_to_tensor, legal_mask, move_to_index


@dataclass(frozen=True)
class PuzzleExample:
    """One supervised training row for "best move from this position"."""
    state: np.ndarray
    target_index: int
    legal_mask: np.ndarray
    fen_key: str
    fen: str
    best_move_uci: str
    puzzle_id: str


def _extract_best_move_uci(row: dict[str, str]) -> str | None:
    """Read first move from Lichess-style `Moves` column."""
    moves = (row.get("Moves") or row.get("moves") or "").strip()
    if not moves:
        return None
    parts = moves.split()
    return parts[0] if parts else None


def _extract_fen(row: dict[str, str]) -> str | None:
    """Read FEN from mixed-case CSV headers (`FEN` or `fen`)."""
    fen = (row.get("FEN") or row.get("fen") or "").strip()
    return fen or None


def _canonical_fen_key(board: chess.Board) -> str:
    """
    Canonical position key for dedupe/splitting.
    Uses only fields relevant to position identity.
    """
    return " ".join(board.fen().split(" ")[:4])


def iter_puzzle_examples(csv_path: str, limit: int | None = None) -> Iterable[PuzzleExample]:
    """
    Load puzzle rows from a CSV file (Lichess-style columns supported).

    Each valid row yields one supervised example:
    - state from FEN
    - target index from the first move in `Moves`
    - legality mask from current board
    """
    kept = 0
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if limit is not None and kept >= limit:
                break

            fen = _extract_fen(row)
            best_uci = _extract_best_move_uci(row)
            if not fen or not best_uci:
                continue

            try:
                board = chess.Board(fen=fen)
                move = chess.Move.from_uci(best_uci)
            except ValueError:
                continue

            if move not in board.legal_moves:
                # Guardrail from plan: ignore illegal-answer entries.
                continue

            target_index = move_to_index(move)
            if target_index < 0 or target_index >= ACTION_SIZE:
                continue

            yield PuzzleExample(
                state=board_to_tensor(board),
                target_index=target_index,
                legal_mask=legal_mask(board),
                fen_key=f"{_canonical_fen_key(board)}|{target_index}",
                fen=fen,
                best_move_uci=best_uci,
                puzzle_id=(row.get("PuzzleId") or row.get("id") or "").strip() or f"row_{kept+1}",
            )
            kept += 1


def load_puzzles(csv_path: str, limit: int | None = None) -> list[PuzzleExample]:
    """Materialize iterator into a list for simpler downstream use."""
    return list(iter_puzzle_examples(csv_path, limit=limit))


def split_train_val(
    examples: list[PuzzleExample], val_ratio: float = 0.1, seed: int = 42
) -> tuple[list[PuzzleExample], list[PuzzleExample]]:
    """
    Split examples by grouped puzzle key to avoid train/val leakage.

    Why grouping: duplicate rows of the same puzzle should not be in both splits.
    """
    if not examples:
        return [], []

    # Group by canonical puzzle key so duplicate rows cannot leak across train/val.
    groups: dict[str, list[PuzzleExample]] = {}
    for ex in examples:
        groups.setdefault(ex.fen_key, []).append(ex)

    keys = list(groups.keys())
    random.Random(seed).shuffle(keys)

    val_group_n = max(1, int(len(keys) * val_ratio)) if len(keys) > 1 else 0
    val_keys = set(keys[:val_group_n])

    train_set: list[PuzzleExample] = []
    val_set: list[PuzzleExample] = []
    for k, rows in groups.items():
        if k in val_keys:
            val_set.extend(rows)
        else:
            train_set.extend(rows)

    return train_set, val_set
