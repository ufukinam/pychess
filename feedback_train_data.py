from __future__ import annotations

"""
Feedback data loader for preference-style move supervision.

Expected JSONL schema (one object per line):
{
  "fen": "<FEN string>",
  "bad_move": "<uci>",
  "good_move": "<uci>",
  "weight": 1.0,              # optional
  "confidence": "medium"      # optional: low|medium|high
}
"""

import json
import random
from dataclasses import dataclass

import chess
import numpy as np

from encode import board_to_tensor, move_to_index


CONFIDENCE_TO_WEIGHT = {
    "low": 0.6,
    "medium": 1.0,
    "high": 1.4,
}


@dataclass
class FeedbackSample:
    state: np.ndarray
    good_idx: int
    bad_idx: int
    weight: float


class FeedbackBuffer:
    def __init__(self):
        self.samples: list[FeedbackSample] = []

    def add_many(self, items: list[FeedbackSample]):
        self.samples.extend(items)

    def sample(self, batch_size: int):
        batch = random.sample(self.samples, batch_size)
        states = np.stack([b.state for b in batch], axis=0).astype(np.float32)
        good = np.array([b.good_idx for b in batch], dtype=np.int64)
        bad = np.array([b.bad_idx for b in batch], dtype=np.int64)
        weights = np.array([b.weight for b in batch], dtype=np.float32)
        return states, good, bad, weights

    def __len__(self):
        return len(self.samples)


def _line_to_sample(obj: dict) -> FeedbackSample | None:
    fen = obj.get("fen")
    bad_uci = obj.get("bad_move")
    good_uci = obj.get("good_move")
    if not fen or not bad_uci or not good_uci:
        return None

    try:
        board = chess.Board(fen)
        bad_move = chess.Move.from_uci(str(bad_uci))
        good_move = chess.Move.from_uci(str(good_uci))
    except Exception:
        return None

    legal = set(board.legal_moves)
    if bad_move not in legal or good_move not in legal:
        return None
    if bad_move == good_move:
        return None

    raw_weight = obj.get("weight")
    conf = str(obj.get("confidence", "")).strip().lower()
    if raw_weight is None:
        weight = CONFIDENCE_TO_WEIGHT.get(conf, 1.0)
    else:
        try:
            weight = float(raw_weight)
        except Exception:
            weight = 1.0
    weight = max(0.05, min(5.0, weight))

    return FeedbackSample(
        state=board_to_tensor(board),
        good_idx=move_to_index(good_move),
        bad_idx=move_to_index(bad_move),
        weight=weight,
    )


def load_feedback_jsonl(path: str, max_samples: int | None = None) -> tuple[list[FeedbackSample], int]:
    """
    Returns:
    - parsed samples
    - rejected line count
    """
    samples: list[FeedbackSample] = []
    rejected = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                rejected += 1
                continue

            sample = _line_to_sample(obj)
            if sample is None:
                rejected += 1
                continue

            samples.append(sample)
            if max_samples is not None and len(samples) >= max_samples:
                break

    return samples, rejected
