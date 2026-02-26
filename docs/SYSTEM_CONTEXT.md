# System Context (AI Memory File)

This file is a high-level technical memory for future sessions.

## 1. Project Mission

This repository trains and evaluates a chess policy-value model using two complementary data sources:

1. Puzzle-supervised learning (predict best move from curated tactical positions).
2. Self-play reinforcement learning (generate trajectories with MCTS, then optimize policy and value targets).

It also includes GUI/CLI tooling for:

- training orchestration,
- game playback,
- human-vs-model play,
- model-vs-model comparison,
- human feedback collection.

## 2. End-to-End Dataflows

### 2.1 Puzzle pretraining path

1. Input puzzle CSV (`puzzles.py`) or prebuilt NPZ cache (`build_puzzle_cache.py`).
2. Convert each board state into tensor planes (`encode.board_to_tensor`).
3. Train policy head with legality-masked logits (`train_puzzles.py`, `puzzle_train_data.py`).
4. Evaluate with top-1/top-5 and legality metrics (`puzzle_train_eval.py`).
5. Save:
   - `checkpoint_puzzle_latest.pt`
   - `checkpoint_puzzle_best.pt`
   - optional best-validation PGN samples.

### 2.2 Self-play RL path

1. Initialize network (`net.AlphaZeroNet`) from:
   - latest self-play checkpoint or
   - puzzle checkpoint (`--prefer_puzzle_init`).
2. Generate games with MCTS (`selfplay.play_self_game` + `mcts.mcts_policy_and_action`).
3. Store `(state, pi, v)` samples in replay buffer (`selfplay_train_core.ReplayBuffer`).
4. Run minibatch updates (`train_step` or `train_step_with_feedback`).
5. Gate candidate net against previous snapshot (`eval.eval_candidate_vs_baseline`).
6. Save accepted checkpoint (`checkpoint_latest.pt`).
7. Periodically evaluate vs random baseline (`eval.eval_net_vs_random`).

### 2.3 Feedback path

1. Create candidate rows from PGNs (`generate_feedback_candidates.py`) or mark manually (`mark_bad_move.py`, GUI dialogs).
2. Store rows in `feedback.jsonl` with `fen`, `bad_move`, `good_move`, optional weight/confidence.
3. Load validated rows (`feedback_train_data.load_feedback_jsonl`).
4. Apply ranking loss term during RL updates (`train_step_with_feedback`).

### 2.4 Human and model gameplay path

- `play_vs_model.py`: human vs model GUI with optional replay/PGN export.
- `model_vs_model.py`: checkpoint-vs-checkpoint GUI for comparison and labeling.
- `pgn_viewer.py`: PGN navigation and feedback annotation.

## 3. Module Interaction Map

Core architecture dependencies:

- `encode.py`
  - Provides board tensor contract and action-index contract for all major modules.
- `net.py`
  - Consumes encoded tensors; outputs policy logits and scalar value.
- `mcts.py`
  - Calls net inference and legality masking; produces improved policy targets and move selection.

Pipeline modules:

- `selfplay.py` -> uses `mcts.py` + `encode.py` + `env.py`
- `selfplay_train_core.py` -> replay sampling and update steps
- `train.py` -> orchestrates RL iterations and gating via `eval.py`
- `puzzles.py` + `puzzle_train_data.py` + `puzzle_train_eval.py` -> supervised puzzle training stack
- `train_puzzles.py` -> orchestrates puzzle stack
- `build_puzzle_cache.py` -> preprocesses CSV into shard format consumed by puzzle training

Interfaces:

- `chess_interface.py` -> CLI dispatcher over script entry points
- `chess_gui.py` -> GUI front-end that shells to `chess_interface.py`
- `chess_board_base.py` -> shared board/capture renderer helpers
- `play_vs_model.py`, `model_vs_model.py`, `pgn_viewer.py` -> user interaction surfaces

## 4. Artifact Lifecycle and Contracts

### 4.1 Encoded state/action contract

- Action size: `ACTION_SIZE = 20480` (`64 * 64 * 5`)
- Input channels: `IN_CHANNELS = 67`
- Contract files:
  - `encode.py` (single source of truth)
  - all training/search modules must remain aligned with this.

### 4.2 Replay shards

- Writer: `replay_store.save_shard`
- Keys: `states`, `pis`, `vs`
- Consumer: `replay_store.load_shards_into_buffer` in `train.py`

### 4.3 Puzzle cache shards

- Builder: `build_puzzle_cache.py`
- Train shard keys: `states`, `target_idx`, `legal_masks_packed`
- Val shard keys: train keys plus `fens`, `moves`, `puzzle_ids`
- Consumer: `puzzle_train_data.load_cached_shard` and `load_cached_val_shard_with_meta`

### 4.4 Checkpoints

- Puzzle training:
  - `checkpoint_puzzle_latest.pt`
  - `checkpoint_puzzle_best.pt`
- Self-play training:
  - `checkpoint_latest.pt` (and periodic `checkpoint_XXX.pt`)
- Typical payload keys:
  - `model_state_dict`
  - `optimizer_state_dict`
  - architecture metadata and iteration/epoch info

## 5. Parameter Surfaces That Matter Most

### 5.1 Search/exploration controls

- `num_sims`, `eval_num_sims`
  - Higher improves move quality but increases runtime.
- `temperature`, `temp_floor`, `temp_moves`
  - Governs exploration vs exploitation.
- `dirichlet_eps`, `dirichlet_alpha`
  - Should be nonzero for self-play root exploration.
  - Should be zero for eval/inference consistency.

### 5.2 RL optimization controls

- `lr`, `weight_decay`, `train_batches`, `batch_size`
  - Drive update stability and throughput.
- `replay_maxlen`
  - Balances recency vs diversity.
- `gate_games`, `gate_min_score`
  - Controls acceptance strictness and variance.

### 5.3 Puzzle-training controls

- `batch_size`, `epochs`, `lr`
- `label_smoothing`
- `cache_dir` vs CSV mode
- auto-tuning knobs (`auto_tune_cpu`, thread/batch grids)

### 5.4 Feedback controls

- `feedback_weight`
- `feedback_batch_size`
- `feedback_margin`
- feedback sample quality (`feedback.jsonl` validity and confidence weighting)

## 6. Known Risks and Safe Defaults

Known risks:

1. Eval paths inheriting self-play Dirichlet noise creates unstable metrics and gating decisions.
2. Puzzle optimizer-state transfer into RL init may destabilize early RL adaptation.
3. Current test-suite history test has stale-move false-failure behavior.
4. Running scripts outside venv can fail due torch DLL mismatch.

Safe defaults:

1. Run all commands with venv Python:
   - `.\.venv\Scripts\python.exe ...`
2. Keep self-play exploration noise enabled, but force eval/inference noise off.
3. Treat puzzle checkpoint as weight transfer by default; load optimizer state only when explicitly intended.
4. Prefer cache mode for large puzzle datasets.
5. Use gating and random-baseline eval together; avoid draw-rate-only interpretations.

## 7. Primary Entry Points

Main scripts:

- `train.py`
- `train_puzzles.py`
- `build_puzzle_cache.py`
- `play_vs_model.py`
- `pgn_viewer.py`

Unified control:

- CLI: `chess_interface.py`
- GUI launcher: `chess_gui.py`

## 8. Reproducible Command Baseline

```powershell
.\.venv\Scripts\python.exe chess_interface.py -h
.\.venv\Scripts\python.exe train_puzzles.py --cache_dir puzzle_cache_lichess --epochs 1 --batch_size 128
.\.venv\Scripts\python.exe train.py --prefer_puzzle_init --iters 1 --games_per_iter 2 --train_batches 2
.\.venv\Scripts\python.exe pgn_viewer.py --load_latest --games_dir games
```

## 9. Future-Session Quick Checks

Before editing core behavior in future sessions:

1. Confirm encoding contract unchanged (`ACTION_SIZE`, `IN_CHANNELS`, legal mask behavior).
2. Confirm search/eval noise policy matches intent.
3. Confirm checkpoint loading policy (weights-only vs weights+optimizer) is explicit.
4. Run test suite from venv and review failures for true regressions vs test bugs.

