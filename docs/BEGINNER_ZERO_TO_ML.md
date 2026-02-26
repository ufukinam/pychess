# Beginner Guide: From Zero to Productive Changes

Audience: developer with no prior ML/DL background  
Goal: understand this project well enough to make safe, confident edits.

## 1. What This Project Does

At a high level, this project teaches a chess AI to choose moves.

It combines:

1. Chess rules (`python-chess`)
2. A neural network (`torch`)
3. Search (`MCTS`) that improves raw network suggestions

The AI learns from:

1. Puzzle examples (supervised learning)
2. Self-play games (reinforcement learning style targets)
3. Optional human feedback rows (`bad move` vs `good move`)

## 2. Core ML Concepts in Simple Language

### 2.1 Model

A model is a function that takes numbers and outputs predictions.

In this project, input numbers represent a chess board.

Output has two parts:

1. Policy: scores for all possible move indices.
2. Value: estimated game outcome from current side perspective (`-1` to `+1`).

### 2.2 Training

Training means repeatedly:

1. show examples,
2. compare model prediction vs target,
3. adjust model weights to reduce error.

### 2.3 Loss

Loss is a single number that tells "how wrong" the model is.

Common losses here:

1. Cross-entropy (policy classification)
2. MSE (value regression)
3. Ranking margin loss (optional feedback preference)

### 2.4 Optimization

An optimizer (AdamW here) updates model weights based on gradients.

Important knobs:

- learning rate (`lr`)
- weight decay
- batch size

### 2.5 Why search (MCTS) is needed

Neural network output alone can be noisy.

MCTS runs many simulations from a position to refine move choice.
Self-play also uses MCTS visit counts as stronger policy targets.

## 3. Libraries Used and Why

- `python-chess`
  - legal moves, board state, PGN/FEN parsing
- `torch`
  - neural net, backprop, optimizer, tensors, GPU support
- `numpy`
  - array manipulation, storage, masks, shard serialization
- `tensorboard`
  - training/eval metric logging
- `tkinter`
  - desktop GUIs for control and game visualization

## 4. Mental Model of the Full Workflow

### 4.1 Puzzle-first workflow (common startup)

1. Build cache from puzzle CSV (optional but recommended for big datasets).
2. Train model to predict best puzzle move.
3. Save best puzzle checkpoint.

### 4.2 Self-play refinement workflow

1. Start from puzzle checkpoint (`--prefer_puzzle_init`).
2. Generate self-play games with MCTS.
3. Train on replay samples.
4. Gate against previous model.
5. Keep accepted improvements.

### 4.3 Human feedback workflow

1. Generate or mark bad-move candidates from PGN.
2. Fill `good_move`.
3. Train with ranking loss mixed into replay loss.

## 5. File Tour in Recommended Learning Order

### Step 1: Encoding contract

- `encode.py`
  - `move_to_index` and `action_to_move`
  - `legal_mask`
  - `board_to_tensor`

Why first: every other module depends on this representation.

### Step 2: Model

- `net.py`
  - `AlphaZeroNet`
  - policy head + value head

### Step 3: Search

- `mcts.py`
  - `Node`
  - `mcts_run`
  - `mcts_policy_and_action`

### Step 4: Data generation

- `selfplay.py`
  - `play_self_game`

### Step 5: Training orchestration

- `selfplay_train_core.py`
- `train.py`

### Step 6: Puzzle supervised pipeline

- `puzzles.py`
- `puzzle_train_data.py`
- `puzzle_train_eval.py`
- `train_puzzles.py`

### Step 7: Operations and UI

- `chess_interface.py` (CLI dispatcher)
- `chess_gui.py` (multi-tab GUI launcher)
- `play_vs_model.py`
- `pgn_viewer.py`
- `model_vs_model.py`

## 6. Key Methods Explained in Plain Language

### 6.1 `encode.board_to_tensor(board, history=None)`

Converts board state into fixed-size numeric planes (`67 x 8 x 8`).

What goes in:

1. Piece locations (current + history)
2. Metadata: side to move, castling rights, en passant file, halfmove clock

Why important:

- If you break this mapping, model/search/training all drift out of sync.

### 6.2 `net.AlphaZeroNet.forward(x)`

Input:

- batch of encoded states

Output:

1. policy logits over `ACTION_SIZE`
2. scalar value in `[-1,1]`

Why two heads:

- policy predicts move quality
- value predicts expected game result

### 6.3 `mcts.mcts_policy_and_action(...)`

Runs simulations and returns:

1. policy distribution from visit counts (`pi`)
2. chosen action

Core idea:

- blend model priors + search statistics for better decisions.

### 6.4 `selfplay.play_self_game(...)`

Generates one self-play game and training samples.

Sample tuple shape:

- `(state, pi, v)`

Where:

- `state`: encoded board
- `pi`: improved policy target from MCTS
- `v`: final game result from perspective of side to move at sample time

### 6.5 `train.py` loop

Each iteration:

1. generate games
2. add to replay
3. train mini-batches
4. evaluate and gate
5. checkpoint if accepted

### 6.6 `train_puzzles.py` loop

Each epoch:

1. train policy on puzzle labels with legal masking
2. evaluate metrics
3. save latest and best checkpoints
4. optionally export best solved validation examples as PGN

## 7. Parameter Glossary (What It Means and Why It Matters)

### 7.1 Search and play parameters

- `num_sims`
  - MCTS simulations per move.
  - Higher: stronger but slower.
- `eval_num_sims`
  - Sims used for evaluation/gating.
  - Too low gives noisy comparisons.
- `temperature`, `temp_floor`, `temp_moves`
  - Controls exploration randomness in move selection.

### 7.2 Self-play stop/shaping parameters

- `draw_penalty`
  - Value assigned to draw-like outcomes.
- `no_progress_limit`
  - Stops long no-capture/no-pawn-progress sequences.
- `repeat2_penalty`, `stop_on_repeat2`
  - Handles repeated-position loops.
- `use_material_shaping`, `material_scale`, `exchange_scale`
  - Adds heuristic reward shaping (can help or bias learning).

### 7.3 Optimization parameters

- `lr`
  - Step size for weight updates.
  - Too high: unstable; too low: very slow.
- `batch_size`
  - Number of samples per update.
  - Higher: smoother gradients, more memory.
- `train_batches`
  - Updates per iteration (self-play loop).
- `weight_decay`
  - Regularization to reduce overfitting.

### 7.4 Replay and gating parameters

- `replay_maxlen`
  - Max replay memory.
  - Small favors recency, large favors diversity.
- `gate_games`
  - Games used to accept/reject new model.
- `gate_min_score`
  - Minimum score threshold for acceptance.

### 7.5 Puzzle-training parameters

- `cache_dir`
  - Use cached shards for large-scale speed/reproducibility.
- `label_smoothing`
  - Softens one-hot labels; can reduce overconfidence.
- `overfit_debug_n`
  - Tiny subset sanity check; should overfit quickly if pipeline is healthy.
- `auto_tune_cpu`, `tune_batch_sizes`, `tune_torch_threads`
  - Throughput tuning controls in CPU mode.

### 7.6 Feedback parameters

- `feedback_weight`
  - How strongly preference loss influences training.
- `feedback_batch_size`
  - Number of feedback rows per update.
- `feedback_margin`
  - Desired logit gap between good and bad moves.

## 8. Practical Step-by-Step Commands (Use These First)

Always use venv Python in this workspace:

```powershell
.\.venv\Scripts\python.exe chess_interface.py -h
```

### 8.1 Build puzzle cache

```powershell
.\.venv\Scripts\python.exe build_puzzle_cache.py --puzzles_csv lichess_db_puzzle.csv --out_dir puzzle_cache_lichess --shard_size 2048 --val_ratio 0.1 --seed 42
```

### 8.2 Train puzzle model

```powershell
.\.venv\Scripts\python.exe train_puzzles.py --cache_dir puzzle_cache_lichess --epochs 5 --batch_size 128 --lr 3e-4
```

### 8.3 Start self-play from puzzle checkpoint

```powershell
.\.venv\Scripts\python.exe train.py --prefer_puzzle_init --puzzle_checkpoint checkpoint_puzzle_best.pt --iters 3 --games_per_iter 10 --train_batches 16
```

### 8.4 Play against model

```powershell
.\.venv\Scripts\python.exe play_vs_model.py --checkpoint checkpoint_latest.pt --num_sims 60 --save_training_samples
```

### 8.5 View PGN

```powershell
.\.venv\Scripts\python.exe pgn_viewer.py --load_latest --games_dir games
```

### 8.6 Use unified CLI and GUI

```powershell
.\.venv\Scripts\python.exe chess_interface.py train-puzzles -h
.\.venv\Scripts\python.exe chess_gui.py
```

## 9. Safe Change Playbook

If you change these files, run these checks:

1. `encode.py` / `mcts.py` / `net.py`
  - Run unit tests:
  - `.\.venv\Scripts\python.exe tests.py`
  - Run tiny smoke train:
  - `.\.venv\Scripts\python.exe train_puzzles.py --cache_dir puzzle_cache_lichess --epochs 1 --batch_size 64`
2. `train.py` or replay logic
  - Run 1-iteration smoke loop with low sims and batches.
3. UI files (`play_vs_model.py`, `pgn_viewer.py`, `model_vs_model.py`, `chess_gui.py`)
  - Verify launch and one basic interaction path.
4. CLI dispatcher (`chess_interface.py`)
  - Validate each affected subcommand with `-h` and one smoke invocation.

## 10. Common Failure Modes and Troubleshooting

### Problem: `python tests.py` fails importing torch

Cause:

- wrong interpreter/environment.

Fix:

- run with:
  - `.\.venv\Scripts\python.exe tests.py`

### Problem: Evaluation scores look unstable between runs

Possible causes:

1. Eval path using Dirichlet exploration noise.
2. Too few eval games/sims.

Fix:

1. Ensure eval/inference calls use `dirichlet_eps=0.0`.
2. Increase `gate_games` and/or `eval_num_sims`.

### Problem: Training starts unstable after puzzle-init

Possible cause:

- loading optimizer state from puzzle checkpoint into RL loop.

Fix:

- use weights-only transfer policy for puzzle-init unless intentionally experimenting with optimizer transfer.

### Problem: Puzzle training is too slow

Possible causes:

1. CSV mode on very large dataset.
2. CPU config not tuned.

Fix:

1. Build and train from cache shards.
2. Use CPU auto-tune options.

### Problem: Model predicts illegal moves

Context:

- raw logits can include illegal actions because action space is fixed-size.

Fix:

- ensure legal-mask application is active in training and evaluation code paths.

## 11. What "Confidently Make Changes" Means Here

Before committing meaningful changes, you should be able to explain:

1. How board encoding maps to network input.
2. How policy/value outputs are consumed by MCTS.
3. How replay and puzzle data are built and validated.
4. Which parameters affect strength vs speed vs stability.
5. Which scripts to run to verify your exact change.

