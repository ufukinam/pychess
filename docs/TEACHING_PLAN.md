# PyChess Teaching Plan (Why -> How -> Code)

## Goal
Teach this project end-to-end so you can:
- rebuild a minimal version from scratch,
- explain each pipeline confidently,
- tune performance with evidence,
- extend the code safely.

This plan is organized in implementation order and each module includes:
- Why
- How
- Code
- Practice
- Success Check

---

## Module 0 - Environment and Orientation

### Why
You need stable run commands and a map of workflows before touching internals.

### How
1. Learn the unified entrypoint and GUI entrypoint.
2. Understand how each tab/subcommand maps to scripts.
3. Run tiny smoke commands.

### Code
- `README.md`
- `chess_interface.py`
- `chess_gui.py`

### Practice
```powershell
python chess_interface.py -h
python chess_interface.py train-selfplay -h
python chess_interface.py train-puzzles -h
python chess_interface.py build-puzzle-cache -h
python chess_interface.py generate-puzzles -h
python chess_interface.py play-vs-model -h
python chess_interface.py pgn-viewer -h
```

### Success Check
You can explain every tab/subcommand and which script it launches.

---

## Module 1 - Encoding (State/Action Interface)

### Why
All model/search/training logic depends on consistent numeric representations.

### How
1. Understand action space size: `64 * 64 * 5`.
2. Understand move-index mapping and inverse.
3. Understand legal move mask and board tensor planes.

### Code
- `encode.py`

### Practice
1. Take a few legal moves from an opening position.
2. Convert move -> index -> move and verify round-trip.
3. Print tensor shape and inspect nonzero planes.

### Success Check
You can derive the action index formula and explain all tensor planes.

---

## Module 2 - Network Architecture

### Why
Policy/value outputs are the model side of AlphaZero-style training.

### How
1. Trace residual trunk.
2. Trace policy head to `ACTION_SIZE`.
3. Trace value head to scalar `[-1, 1]`.

### Code
- `net.py`

### Practice
Run a forward pass on random input and verify shapes.

### Success Check
You can explain why both heads exist and where each is consumed.

---

## Module 3 - MCTS Integration

### Why
Raw logits are weak move selectors; search improves decision quality.

### How
1. Follow node expansion and priors.
2. Follow selection and backup perspective/sign.
3. Follow legality masking before policy usage.

### Code
- `mcts.py`

### Practice
Trace one move decision from root to selected action.

### Success Check
You can explain how value and priors interact in search.

---

## Module 4 - Self-Play Data Generation

### Why
RL training needs `(state, policy_target, value_target)` from self-play.

### How
1. Understand simulation schedule and temperature.
2. Understand anti-loop controls and shaping.
3. Understand trajectory to sample conversion.

### Code
- `selfplay.py` (`play_self_game`)
- `env.py`

### Practice
Run 1-2 short games, inspect PGN and generated sample tuple logic.

### Success Check
You can justify loop-control heuristics and their training impact.

---

## Module 5 - RL Training Loop and Replay

### Why
Samples must be persisted, replayed, and optimized iteratively.

### How
1. Understand replay buffer load/save path.
2. Understand iteration loop: self-play -> train -> eval -> checkpoint.
3. Understand startup checkpoint transfer behavior.

### Code
- `train.py`
- `selfplay_train_core.py`
- `replay_store.py`
- `eval.py`

### Practice
Run tiny iteration counts and interpret loss/eval outputs.

### Success Check
You can draw the complete loop and explain each checkpoint artifact.

---

## Module 6 - Puzzle Supervised Training

### Why
Puzzle pretraining gives stronger policy initialization than RL cold start.

### How
1. Parse puzzle rows and validate targets.
2. Train/eval with legality-masked policy loss.
3. Understand checkpoint and best-model logic.

### Code
- `puzzles.py`
- `train_puzzles.py`
- `puzzle_train_data.py`
- `puzzle_train_eval.py`

### Practice
Run one short training from CSV and one from cache, compare metrics.

### Success Check
You can explain `val_top1`, `val_top5`, `raw_legality_rate`, `masked_legality_rate`.

---

## Module 7 - Cache Pipeline for Scale

### Why
Repeated CSV parsing is expensive for large datasets.

### How
1. Build sharded NPZ cache with deterministic split.
2. Understand train/val shard schema.
3. Tune workers/compression/shard size tradeoffs.

### Code
- `build_puzzle_cache.py`
- `puzzle_cache_*/manifest.json`

### Practice
```powershell
python build_puzzle_cache.py --puzzles_csv lichess_db_puzzle.csv --out_dir puzzle_cache_smoke --limit 2000 --workers 2 --compression none --clean_out_dir
python build_puzzle_cache.py --puzzles_csv lichess_db_puzzle.csv --out_dir puzzle_cache_lichess --workers 0 --compression compressed
```

### Success Check
You can explain when to use `compression=none` and how to choose shard settings.

---

## Module 8 - Throughput and Performance Tuning

### Why
CPU usage percentage alone is not a valid optimization target.

### How
1. Use timing logs (`[TrainTiming]`, `[EpochTiming]`).
2. Measure throughput (`samples/s`) across configs.
3. Use auto tuner to select best `(batch_size, torch_threads)`.

### Code
- `train_puzzles.py`
- `puzzle_train_data.py`

### Practice
```powershell
python train_puzzles.py --cache_dir puzzle_cache_smoke --auto_tune_cpu --tune_only --tune_batch_sizes 128,256,512 --tune_torch_threads 4,6,8 --tune_max_batches 120
python train_puzzles.py --cache_dir puzzle_cache_lichess --auto_tune_cpu --tune_batch_sizes 128,256,512 --tune_torch_threads 4,6,8 --tune_max_batches 120 --epochs 5 --lr 3e-4
```

### Success Check
You can report best config by throughput and justify it.

---

## Module 9 - Interface and UX Control Layer

### Why
Reliable operations require consistent command dispatch and GUI parity.

### How
1. Trace CLI subcommands to script args.
2. Trace GUI form fields to CLI args.
3. Verify parity with same workflow from both interfaces.

### Code
- `chess_interface.py`
- `chess_gui.py`

### Practice
Run one workflow from GUI and CLI with identical settings and compare logs.

### Success Check
You can add one new argument script -> interface -> GUI without mismatch.

---

## Module 10 - Testing and Debug Strategy

### Why
Speed changes and refactors need correctness guardrails.

### How
1. Use unit checks for encoding and MCTS invariants.
2. Use overfit mode sanity checks.
3. Use regression checklist after optimization changes.

### Code
- `tests.py`
- `train_puzzles.py` (`--overfit_debug_n`)

### Practice
```powershell
python tests.py
python train_puzzles.py --cache_dir puzzle_cache_smoke --epochs 1 --overfit_debug_n 128
```

### Success Check
You can separate true bug vs performance variance using tests + logs.

---

## Build-From-Scratch Milestones
1. Recreate encoding module.
2. Recreate minimal net.
3. Recreate MCTS core.
4. Recreate self-play sample generation.
5. Recreate replay + train step.
6. Recreate iterative self-play trainer.
7. Recreate puzzle CSV trainer.
8. Recreate cache builder + cached trainer.
9. Recreate interface dispatcher.
10. Recreate GUI runner.
11. Recreate timing and auto-tune tooling.
12. Recreate tests and debugging playbook.

Do not move to next milestone until current milestone passes its checks.

---

## Default Practical Settings
- Puzzle LR: `3e-4`
- CPU tuning grid: batch `128,256,512`; threads `4,6,8`
- Use cache mode for large datasets
- `progress_every_batches=100` as default diagnostic frequency

