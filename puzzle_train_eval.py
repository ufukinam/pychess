from __future__ import annotations

import os

import chess
import chess.pgn
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from net import AlphaZeroNet
from puzzle_train_data import (
    load_cached_shard,
    load_cached_val_shard_with_meta,
    mask_illegal_logits,
)
from puzzles import PuzzleExample


def evaluate_puzzle_validation(
    net: AlphaZeroNet, loader: DataLoader, device: str
) -> dict[str, float]:
    net.eval()
    total = 0
    loss_sum = 0.0
    top1_hits = 0
    top5_hits = 0
    raw_legal_pred_hits = 0
    masked_legal_pred_hits = 0

    with torch.inference_mode():
        for x, target_idx, legal_masks in loader:
            x = x.to(device)
            target_idx = target_idx.to(device)
            legal_masks = legal_masks.to(device)

            logits, _ = net(x)
            masked_logits = mask_illegal_logits(logits, legal_masks)
            loss = F.cross_entropy(masked_logits, target_idx, reduction="sum")
            loss_sum += float(loss.item())

            top1 = torch.argmax(masked_logits, dim=1)
            top5 = torch.topk(
                masked_logits, k=min(5, masked_logits.shape[1]), dim=1
            ).indices

            top1_hits += int((top1 == target_idx).sum().item())
            top5_hits += int((top5 == target_idx.unsqueeze(1)).any(dim=1).sum().item())

            raw_top1 = torch.argmax(logits, dim=1)
            raw_legal_pred_hits += int(
                legal_masks.gather(1, raw_top1.unsqueeze(1)).squeeze(1).sum().item()
            )
            masked_legal_pred_hits += int(
                legal_masks.gather(1, top1.unsqueeze(1)).squeeze(1).sum().item()
            )
            total += x.size(0)

    if total == 0:
        return {
            "val_loss": 0.0,
            "val_top1": 0.0,
            "val_top5": 0.0,
            "raw_legality_rate": 0.0,
            "masked_legality_rate": 0.0,
            "legality_rate": 0.0,
        }

    return {
        "val_loss": loss_sum / total,
        "val_top1": top1_hits / total,
        "val_top5": top5_hits / total,
        "raw_legality_rate": raw_legal_pred_hits / total,
        "masked_legality_rate": masked_legal_pred_hits / total,
        "legality_rate": raw_legal_pred_hits / total,
    }


def evaluate_puzzle_validation_from_shards(
    net: AlphaZeroNet,
    shard_paths: list[str],
    device: str,
    batch_size: int,
    progress_every_shards: int = 1,
) -> dict[str, float]:
    if not shard_paths:
        return {
            "val_loss": 0.0,
            "val_top1": 0.0,
            "val_top5": 0.0,
            "raw_legality_rate": 0.0,
            "masked_legality_rate": 0.0,
            "legality_rate": 0.0,
        }

    net.eval()
    total = 0
    loss_sum = 0.0
    top1_hits = 0
    top5_hits = 0
    raw_legal_pred_hits = 0
    masked_legal_pred_hits = 0

    with torch.inference_mode():
        total_shards = len(shard_paths)
        bs = max(1, int(batch_size))
        for shard_idx, path in enumerate(shard_paths, start=1):
            states, target_idx, legal_masks = load_cached_shard(path)
            shard_n = int(states.shape[0])
            for start in range(0, shard_n, bs):
                end = start + bs
                x = torch.from_numpy(states[start:end]).float().to(device)
                t = torch.from_numpy(target_idx[start:end]).long().to(device)
                mask = torch.from_numpy(legal_masks[start:end]).to(device)

                logits, _ = net(x)
                masked_logits = mask_illegal_logits(logits, mask)
                loss = F.cross_entropy(masked_logits, t, reduction="sum")
                loss_sum += float(loss.item())

                top1 = torch.argmax(masked_logits, dim=1)
                top5 = torch.topk(
                    masked_logits, k=min(5, masked_logits.shape[1]), dim=1
                ).indices
                top1_hits += int((top1 == t).sum().item())
                top5_hits += int((top5 == t.unsqueeze(1)).any(dim=1).sum().item())

                raw_top1 = torch.argmax(logits, dim=1)
                raw_legal_pred_hits += int(
                    mask.gather(1, raw_top1.unsqueeze(1)).squeeze(1).sum().item()
                )
                masked_legal_pred_hits += int(
                    mask.gather(1, top1.unsqueeze(1)).squeeze(1).sum().item()
                )
                total += x.size(0)
            if progress_every_shards > 0 and (
                shard_idx % progress_every_shards == 0 or shard_idx == total_shards
            ):
                print(f"[Val] shard {shard_idx}/{total_shards} processed")

    if total == 0:
        return {
            "val_loss": 0.0,
            "val_top1": 0.0,
            "val_top5": 0.0,
            "raw_legality_rate": 0.0,
            "masked_legality_rate": 0.0,
            "legality_rate": 0.0,
        }
    return {
        "val_loss": loss_sum / total,
        "val_top1": top1_hits / total,
        "val_top5": top5_hits / total,
        "raw_legality_rate": raw_legal_pred_hits / total,
        "masked_legality_rate": masked_legal_pred_hits / total,
        "legality_rate": raw_legal_pred_hits / total,
    }


def save_best_validation_pgns(
    net: AlphaZeroNet,
    val_examples: list[PuzzleExample],
    device: str,
    out_dir: str,
    epoch: int,
    max_games: int = 20,
) -> tuple[str | None, int]:
    if not val_examples or max_games <= 0:
        return None, 0

    net.eval()
    solved: list[tuple[float, PuzzleExample]] = []

    with torch.inference_mode():
        for ex in val_examples:
            x = torch.from_numpy(ex.state).unsqueeze(0).to(device)
            legal = torch.from_numpy(ex.legal_mask.astype(np.bool_)).unsqueeze(0).to(device)
            logits, _ = net(x)
            masked_logits = mask_illegal_logits(logits, legal)
            probs = torch.softmax(masked_logits, dim=1).squeeze(0)
            pred = int(torch.argmax(masked_logits, dim=1).item())
            if pred == ex.target_index:
                solved.append((float(probs[ex.target_index].item()), ex))

    if not solved:
        return None, 0

    solved.sort(key=lambda t: t[0], reverse=True)
    picked = solved[:max_games]

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"best_epoch_{epoch:03d}.pgn")
    with open(out_path, "w", encoding="utf-8") as f:
        for conf, ex in picked:
            board = chess.Board(ex.fen)
            move = chess.Move.from_uci(ex.best_move_uci)
            if move not in board.legal_moves:
                continue

            game = chess.pgn.Game()
            game.setup(board)
            game.headers["Event"] = "PuzzleValidationBest"
            game.headers["Site"] = "Local"
            game.headers["Round"] = "-"
            game.headers["White"] = "PuzzleSide"
            game.headers["Black"] = "PuzzleOpp"
            game.headers["Result"] = "*"
            game.headers["SetUp"] = "1"
            game.headers["FEN"] = ex.fen
            game.headers["PuzzleId"] = ex.puzzle_id
            game.headers["TargetMove"] = ex.best_move_uci
            game.headers["Confidence"] = f"{conf:.4f}"

            game.add_variation(move)
            print(game, file=f, end="\n\n")

    return out_path, len(picked)


def save_best_validation_pgns_from_shards(
    net: AlphaZeroNet,
    val_shard_paths: list[str],
    device: str,
    out_dir: str,
    epoch: int,
    max_games: int = 20,
) -> tuple[str | None, int]:
    if not val_shard_paths or max_games <= 0:
        return None, 0

    net.eval()
    solved: list[tuple[float, str, str, str]] = []
    with torch.inference_mode():
        for path in val_shard_paths:
            states, target_idx, legal_masks, fens, moves, puzzle_ids = load_cached_val_shard_with_meta(path)
            n = int(states.shape[0])
            if len(fens) != n or len(moves) != n or len(puzzle_ids) != n:
                continue

            x = torch.from_numpy(states).to(device)
            t = torch.from_numpy(target_idx).to(device)
            mask = torch.from_numpy(legal_masks).to(device)
            logits, _ = net(x)
            masked_logits = mask_illegal_logits(logits, mask)
            probs = torch.softmax(masked_logits, dim=1)
            pred = torch.argmax(masked_logits, dim=1)
            good = (pred == t).cpu().numpy()
            conf = probs.gather(1, t.unsqueeze(1)).squeeze(1).cpu().numpy()

            for i in range(n):
                if bool(good[i]):
                    solved.append((float(conf[i]), str(fens[i]), str(moves[i]), str(puzzle_ids[i])))

    if not solved:
        return None, 0

    solved.sort(key=lambda t: t[0], reverse=True)
    picked = solved[:max_games]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"best_epoch_{epoch:03d}.pgn")
    with open(out_path, "w", encoding="utf-8") as f:
        for conf, fen, best_move_uci, puzzle_id in picked:
            board = chess.Board(fen)
            move = chess.Move.from_uci(best_move_uci)
            if move not in board.legal_moves:
                continue
            game = chess.pgn.Game()
            game.setup(board)
            game.headers["Event"] = "PuzzleValidationBest"
            game.headers["Site"] = "Local"
            game.headers["Round"] = "-"
            game.headers["White"] = "PuzzleSide"
            game.headers["Black"] = "PuzzleOpp"
            game.headers["Result"] = "*"
            game.headers["SetUp"] = "1"
            game.headers["FEN"] = fen
            game.headers["PuzzleId"] = puzzle_id
            game.headers["TargetMove"] = best_move_uci
            game.headers["Confidence"] = f"{conf:.4f}"
            game.add_variation(move)
            print(game, file=f, end="\n\n")
    return out_path, len(picked)
