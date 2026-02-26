# PyChess Training Toolkit

Unified toolkit for:
- self-play RL training
- puzzle pretraining
- puzzle cache building from large CSVs
- synthetic puzzle generation
- human-vs-model and model-vs-model GUI play
- PGN viewing and feedback labeling
- CLI + GUI orchestration

## Environment

Use the project venv for all commands:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Main Entrypoints

- Unified CLI: `chess_interface.py`
- Visual control panel: `chess_gui.py`

## Documentation (Current On-Disk Set)

- `docs/BEGINNER_ZERO_TO_ML.md`: zero-prerequisite onboarding guide
- `docs/SYSTEM_CONTEXT.md`: architecture and dataflow memory file
- `docs/CODEBASE_AUDIT.md`: integrity findings + patch-ready risk specs
- `docs/FILE_CATALOG_TRACKED.md`: tracked-file catalog with roles and dependencies

## Quick Start

```powershell
.\.venv\Scripts\python.exe chess_interface.py -h
.\.venv\Scripts\python.exe chess_gui.py
```

Subcommand help:

```powershell
.\.venv\Scripts\python.exe chess_interface.py train-selfplay -h
.\.venv\Scripts\python.exe chess_interface.py train-puzzles -h
.\.venv\Scripts\python.exe chess_interface.py build-puzzle-cache -h
.\.venv\Scripts\python.exe chess_interface.py generate-puzzles -h
.\.venv\Scripts\python.exe chess_interface.py play-vs-model -h
.\.venv\Scripts\python.exe chess_interface.py pgn-viewer -h
.\.venv\Scripts\python.exe chess_interface.py model-vs-model -h
.\.venv\Scripts\python.exe chess_interface.py generate-feedback-candidates -h
```

## Core Workflows

### 1) Build Puzzle Cache (recommended for large datasets)

```powershell
.\.venv\Scripts\python.exe chess_interface.py build-puzzle-cache `
  --puzzles_csv "lichess_db_puzzle.csv" `
  --out_dir "puzzle_cache_lichess" `
  --shard_size 2048 `
  --val_ratio 0.1 `
  --seed 42 `
  --workers 0 `
  --compression compressed `
  --clean_out_dir
```

Cache schema:
- train shards: `states`, `target_idx`, `legal_masks_packed`
- val shards: train keys + `fens`, `moves`, `puzzle_ids`

### 2) Puzzle Pretraining

From cache:

```powershell
.\.venv\Scripts\python.exe chess_interface.py train-puzzles `
  --cache_dir "puzzle_cache_lichess" `
  --epochs 10 `
  --batch_size 128 `
  --lr 3e-4 `
  --progress_every_batches 100
```

From CSV:

```powershell
.\.venv\Scripts\python.exe chess_interface.py train-puzzles `
  --puzzles_csv "puzzles_synthetic.csv" `
  --epochs 5 `
  --batch_size 128 `
  --lr 3e-4
```

Primary outputs:
- `checkpoint_puzzle_latest.pt`
- `checkpoint_puzzle_best.pt`
- optional PGN exports in `puzzle_games/`

### 3) Self-Play RL Training

Weights-only puzzle transfer (default):

```powershell
.\.venv\Scripts\python.exe chess_interface.py train-selfplay `
  --prefer_puzzle_init `
  --puzzle_checkpoint "checkpoint_puzzle_best.pt"
```

If you explicitly want optimizer restore from puzzle checkpoint:

```powershell
.\.venv\Scripts\python.exe chess_interface.py train-selfplay `
  --prefer_puzzle_init `
  --puzzle_checkpoint "checkpoint_puzzle_best.pt" `
  --load_optimizer_from_puzzle_init
```

### 4) Play, Compare, and Inspect Games

Human vs model:

```powershell
.\.venv\Scripts\python.exe chess_interface.py play-vs-model `
  --checkpoint "checkpoint_latest.pt" `
  --num_sims 80 `
  --save_training_samples
```

Model vs model GUI:

```powershell
.\.venv\Scripts\python.exe chess_interface.py model-vs-model `
  --num_sims 50 `
  --move_delay_ms 400 `
  --pgn_dir "model_games"
```

PGN viewer:

```powershell
.\.venv\Scripts\python.exe chess_interface.py pgn-viewer --load_latest --games_dir "games"
```

### 5) Feedback Candidate Generation

```powershell
.\.venv\Scripts\python.exe chess_interface.py generate-feedback-candidates `
  --pgn_glob "model_games/*.pgn" `
  --out "feedback_candidates.jsonl" `
  --side both `
  --max_legal_moves 20
```

## Visual Control Panel

Launch:

```powershell
.\.venv\Scripts\python.exe chess_gui.py
```

Tabs:
- Self-Play Train
- Puzzle Train
- Build Puzzle Cache
- Generate Puzzles
- Play vs Model
- Model vs Model
- PGN Viewer
- Feedback Candidates

## Script Entrypoints (Direct)

- `train.py`
- `train_puzzles.py`
- `build_puzzle_cache.py`
- `generate_puzzles.py`
- `play_vs_model.py`
- `model_vs_model.py`
- `pgn_viewer.py`
- `generate_feedback_candidates.py`

## Notes

- Use cache mode for large puzzle datasets.
- Evaluation/inference paths run with Dirichlet root noise disabled for consistency.
- Run tests with:

```powershell
.\.venv\Scripts\python.exe tests.py
```
