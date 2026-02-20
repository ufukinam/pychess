# PyChess Training Toolkit

Unified toolkit for:
- self-play training
- puzzle pretraining
- large puzzle cache building
- synthetic puzzle generation
- PGN viewing
- human vs model play
- visual control panel (no terminal required)

All major workflows can be controlled from one interface:
`chess_interface.py`

If you prefer a visual UI, use:
`chess_gui.py`

## Quick Start

Run the unified interface help:

```bash
python chess_interface.py -h
```

Run the visual control panel:

```bash
python chess_gui.py
```

List subcommand help:

```bash
python chess_interface.py train-puzzles -h
python chess_interface.py train-selfplay -h
python chess_interface.py build-puzzle-cache -h
python chess_interface.py play-vs-model -h
python chess_interface.py pgn-viewer -h
```

## Main Workflows

### 1) Build Puzzle Cache (for big CSV like Lichess)

```bash
python chess_interface.py build-puzzle-cache \
  --puzzles_csv "lichess_db_puzzle.csv" \
  --out_dir "puzzle_cache_lichess" \
  --shard_size 2048 \
  --val_ratio 0.1 \
  --seed 42
```

Optional shard caps:

```bash
python chess_interface.py build-puzzle-cache \
  --puzzles_csv "lichess_db_puzzle.csv" \
  --out_dir "puzzle_cache_small" \
  --shard_size 2048 \
  --max_train_shards 200 \
  --max_val_shards 20
```

### 2) Train on Puzzles

From cache (recommended):

```bash
python chess_interface.py train-puzzles \
  --cache_dir "puzzle_cache_lichess" \
  --epochs 20 \
  --batch_size 128 \
  --lr 3e-4 \
  --progress_every_batches 100
```

From CSV directly:

```bash
python chess_interface.py train-puzzles \
  --puzzles_csv "puzzles_synthetic.csv" \
  --epochs 10 \
  --batch_size 128
```

Outputs:
- `checkpoint_puzzle_latest.pt`
- `checkpoint_puzzle_best.pt`
- best validation PGNs in `puzzle_games/` (configurable)

### 3) Start Self-Play from Puzzle Model

```bash
python chess_interface.py train-selfplay \
  --prefer_puzzle_init \
  --puzzle_checkpoint "checkpoint_puzzle_best.pt"
```

### 4) Play and Inspect Games

Play vs model:

```bash
python chess_interface.py play-vs-model \
  --checkpoint "checkpoint_latest.pt" \
  --num_sims 80 \
  --save_training_samples
```

Open viewer:

```bash
python chess_interface.py pgn-viewer --load_latest
```

or:

```bash
python chess_interface.py pgn-viewer --pgn_path "games/game_xxx.pgn"
```

## Visual Interface (No Terminal Needed)

Launch:

```bash
python chess_gui.py
```

Tabs available:
- Self-Play Train
- Puzzle Train
- Build Puzzle Cache
- Generate Puzzles
- Play vs Model
- PGN Viewer

Features:
- form-based parameter editing
- one-click run for selected workflow
- live output console inside UI
- stop running process button

## Unified Interface: Subcommands

## `train-selfplay`
Runs `train.py`.

Parameters:
- `--init_checkpoint`: primary checkpoint path.
- `--puzzle_checkpoint`: puzzle checkpoint path.
- `--prefer_puzzle_init`: prefer puzzle checkpoint when available.

## `train-puzzles`
Runs `train_puzzles.py`.

Parameters:
- `--cache_dir`: NPZ cache directory (`train_shard_*.npz`, `val_shard_*.npz`).
- `--puzzles_csv`: direct CSV path (if cache not used).
- `--limit`: max examples (0 = unlimited).
- `--batch_size`: training batch size.
- `--epochs`: number of epochs.
- `--lr`: learning rate.
- `--val_ratio`: validation split ratio (CSV mode).
- `--seed`: random seed.
- `--label_smoothing`: CE label smoothing.
- `--overfit_debug_n`: overfit debug sample count.
- `--pgn_dir`: output folder for best validation puzzle PGNs.
- `--pgn_max_games`: max exported PGNs per best epoch.
- `--progress_every_batches`: log interval for training batches.
- `--resume_checkpoint`: puzzle checkpoint to resume from.

Metrics:
- `train_loss`, `val_loss`
- `val_top1`, `val_top5`
- `raw_legality_rate`, `masked_legality_rate`

## `build-puzzle-cache`
Runs `build_puzzle_cache.py`.

Parameters:
- `--puzzles_csv`: source puzzle CSV.
- `--out_dir`: cache output folder.
- `--limit`: optional max read rows.
- `--val_ratio`: validation ratio.
- `--seed`: split seed.
- `--shard_size`: samples per shard.
- `--max_train_shards`: cap train shard count (0 no cap).
- `--max_val_shards`: cap val shard count (0 no cap).

Cache format:
- train shard keys: `states`, `target_idx`, `legal_masks_packed`
- val shard keys: above + `fens`, `moves`, `puzzle_ids`

## `generate-puzzles`
Runs `generate_puzzles.py`.

Parameters:
- `--out`: output CSV path.
- `--count`: number of synthetic puzzles.
- `--seed`: random seed.

## `play-vs-model`
Runs `play_vs_model.py`.

Parameters:
- `--device`: `cpu` or `cuda`.
- `--checkpoint`: model checkpoint path.
- `--num_sims`: MCTS simulations per model move.
- `--play_as_black`: start as black.
- `--save_training_samples`: save human replay shards.
- `--human_replay_dir`: replay output dir.
- `--human_pgn_dir`: PGN output dir.

## `pgn-viewer`
Runs `pgn_viewer.py`.

Parameters:
- `--pgn_path`: PGN file to open immediately.
- `--load_latest`: load latest PGN from games directory.
- `--games_dir`: directory used by `--load_latest`.

## Existing Script Entry Points

You can still run scripts directly:
- `train.py`
- `train_puzzles.py`
- `build_puzzle_cache.py`
- `generate_puzzles.py`
- `play_vs_model.py`
- `pgn_viewer.py`

The unified interface is a convenience layer over these.

## Reliability Tips

- Use puzzle cache for large datasets (fast/reproducible).
- Prefer `checkpoint_puzzle_best.pt` for transfer.
- Monitor `val_top1` + `eval_random/score`, not draw rate alone.
- If cache shards are partially corrupted, trainer skips invalid shards and continues.
