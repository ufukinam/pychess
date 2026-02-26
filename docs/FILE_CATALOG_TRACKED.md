# File Catalog (Tracked Files Only)

Catalog source: `git ls-files`  
Tracked-file count: `36`  
Coverage rule: each tracked file appears exactly once in this document.

## 01. `.gitignore`
- Category: Config/dependency
- Purpose: Defines ignored runtime/generated files (replay shards, caches, runs, PGNs, local env artifacts).
- Key classes/functions or artifact role: Git ignore rules for reproducible repository state.
- Upstream dependencies: Git tooling and project output directory conventions.
- Downstream consumers: Developer workflows, CI cleanliness, commit hygiene.
- Operational notes: Checkpoint `*.pt` is intentionally not fully ignored in current tracked state.

## 02. `README.md`
- Category: Docs
- Purpose: Top-level operational guide for CLI, GUI, and workflow entry points.
- Key classes/functions or artifact role: Command reference and workflow overview.
- Upstream dependencies: Behavior of `chess_interface.py` and script CLIs.
- Downstream consumers: New users and operators.
- Operational notes: Should stay aligned with `chess_interface.py` argument surface.

## 03. `build_puzzle_cache.py`
- Category: Training pipeline
- Purpose: Converts puzzle CSV rows into train/val NPZ shards for fast repeated training.
- Key classes/functions or artifact role:
  - `_ShardWriter`
  - `_process_row`
  - `_is_val_key`
  - `main`
- Upstream dependencies: `encode.py`, `python-chess`, CSV puzzle source.
- Downstream consumers: `train_puzzles.py` cache mode via `puzzle_train_data.py`.
- Operational notes: Supports shard caps, multiprocess conversion, and compressed/uncompressed write modes.

## 04. `checkpoint_puzzle_best.pt`
- Category: Data artifact
- Purpose: Best-performing puzzle-training checkpoint by validation top-1 metric.
- Key classes/functions or artifact role: Serialized model + optimizer payload.
- Upstream dependencies: Produced by `train_puzzles.py`.
- Downstream consumers: Puzzle inference, self-play warm start (`train.py`, `--prefer_puzzle_init`).
- Operational notes: Contains `optimizer_state_dict`; transfer policy should be explicit for RL init.

## 05. `checkpoint_puzzle_best_old.pt`
- Category: Data artifact
- Purpose: Historical puzzle-best checkpoint retained for fallback/comparison.
- Key classes/functions or artifact role: Legacy serialized model artifact.
- Upstream dependencies: Earlier puzzle-training runs.
- Downstream consumers: Manual rollback or comparison workflows.
- Operational notes: Architecture metadata compatibility should be checked before loading.

## 06. `checkpoint_puzzle_latest.pt`
- Category: Data artifact
- Purpose: Latest snapshot from puzzle training loop.
- Key classes/functions or artifact role: Rolling checkpoint with model/optimizer state.
- Upstream dependencies: `train_puzzles.py`.
- Downstream consumers: Resume training, transfer to self-play init.
- Operational notes: May not be the strongest checkpoint; use with metric context.

## 07. `checkpoint_puzzle_latest_old.pt`
- Category: Data artifact
- Purpose: Older latest puzzle checkpoint retained as backup.
- Key classes/functions or artifact role: Historical model artifact.
- Upstream dependencies: Previous training sessions.
- Downstream consumers: Manual restore paths.
- Operational notes: Useful when validating regression across versions.

## 08. `chess_board_base.py`
- Category: Interface/UI
- Purpose: Shared Tkinter board rendering and captured-piece panel helpers.
- Key classes/functions or artifact role:
  - `BoardRenderer`
  - `create_capture_grid`
  - `recompute_captures`
  - `update_capture_display`
- Upstream dependencies: `python-chess`, Tkinter canvas/widgets.
- Downstream consumers: `pgn_viewer.py`, `play_vs_model.py`, `model_vs_model.py`.
- Operational notes: Centralized visual geometry logic reduces duplicate GUI bugs.

## 09. `chess_gui.py`
- Category: Interface/UI
- Purpose: Multi-tab control panel that launches major workflows and streams logs.
- Key classes/functions or artifact role: `ChessControlPanel`.
- Upstream dependencies: Tkinter, subprocess execution of `chess_interface.py`.
- Downstream consumers: Non-terminal users orchestrating training/eval tasks.
- Operational notes: Builds command lines from GUI fields; parity with CLI args is critical.

## 10. `chess_interface.py`
- Category: Interface/UI
- Purpose: Unified CLI dispatcher for self-play, puzzle training, cache building, play/view tools, and feedback generation.
- Key classes/functions or artifact role:
  - `cmd_*` wrappers
  - `_build_parser`
  - `main`
- Upstream dependencies: Script entry points and argument compatibility.
- Downstream consumers: Human operators, `chess_gui.py`.
- Operational notes: Acts as single operational control layer for most workflows.

## 11. `docs/BEGINNER_ZERO_TO_ML.md`
- Category: Docs
- Purpose: Zero-prerequisite onboarding guide for developers new to ML, DL, and this chess codebase.
- Key classes/functions or artifact role:
  - Plain-language ML/DL concepts used by the project
  - Learning-order walkthrough of major project files
  - Parameter glossary with practical effects
  - Step-by-step runbook and troubleshooting guide
- Upstream dependencies: Current behavior and CLI surfaces of training/search/UI scripts.
- Downstream consumers: New developers, reviewers, and future maintainers making first safe changes.
- Operational notes: Should be updated when command examples, defaults, or major workflow steps change.

## 12. `docs/CODEBASE_AUDIT.md`
- Category: Docs
- Purpose: Integrity and risk audit summary focused on model-performance-critical behavior and verification outcomes.
- Key classes/functions or artifact role:
  - Environment integrity snapshot
  - Severity-ranked critical findings
  - Patch-ready remediation specs
  - Validation scenario checklist
- Upstream dependencies: Observed test results, checkpoint/shard inspections, and code-level evidence.
- Downstream consumers: Engineers planning corrective patches and release-readiness checks.
- Operational notes: Should be refreshed after major training-loop, evaluation, or checkpoint-policy changes.

## 13. `docs/SYSTEM_CONTEXT.md`
- Category: Docs
- Purpose: High-level architecture memory file for future AI and developer sessions.
- Key classes/functions or artifact role:
  - End-to-end dataflow (puzzle pretrain, self-play RL, eval/gating, GUI tooling)
  - Module interaction map across encode/net/mcts/train stacks
  - Artifact lifecycle and key parameter surfaces
  - Known risks and recommended safe defaults
- Upstream dependencies: Current module boundaries, artifact schemas, and operational workflows.
- Downstream consumers: Future implementation planning, debugging sessions, and context handoff.
- Operational notes: Must be updated when architecture contracts or training/evaluation policies change.

## 14. `encode.py`
- Category: Core ML
- Purpose: State/action encoding contract shared by model, search, and training.
- Key classes/functions or artifact role:
  - `ACTION_SIZE`, `IN_CHANNELS`
  - `move_to_index`, `action_to_move`, `legal_mask`
  - `board_to_tensor`
  - color-flip augmentation helpers
- Upstream dependencies: `python-chess`, NumPy.
- Downstream consumers: `net.py`, `mcts.py`, self-play and puzzle training pipelines.
- Operational notes: Any change here is high risk; this is the canonical interface layer.

## 15. `env.py`
- Category: Core ML
- Purpose: Thin chess environment wrapper around `python-chess` board state.
- Key classes/functions or artifact role: `ChessEnv`.
- Upstream dependencies: `python-chess`.
- Downstream consumers: `selfplay.py`, `eval.py`.
- Operational notes: Encapsulates terminal/result semantics from White perspective.

## 16. `eval.py`
- Category: Evaluation
- Purpose: Offline model evaluation helpers (vs random and candidate-vs-baseline).
- Key classes/functions or artifact role:
  - `play_vs_random`
  - `play_net_vs_net`
  - `eval_candidate_vs_baseline`
  - `eval_net_vs_random`
- Upstream dependencies: `mcts.py`, `env.py`, `encode.py`.
- Downstream consumers: `train.py` gating and random-baseline evaluation.
- Operational notes: Evaluation behavior must be deterministic enough for stable gate decisions.

## 17. `feedback.jsonl`
- Category: Data artifact
- Purpose: Human-labeled preference feedback rows (bad move vs good move).
- Key classes/functions or artifact role: Input corpus for ranking-style feedback loss.
- Upstream dependencies: `pgn_viewer.py` / `model_vs_model.py` / `mark_bad_move.py` labeling flows.
- Downstream consumers: `feedback_train_data.py`, then `train.py`.
- Operational notes: Row validity depends on legal move and FEN consistency.

## 18. `feedback_train_data.py`
- Category: Training pipeline
- Purpose: Loads and validates JSONL preference feedback into training tensors.
- Key classes/functions or artifact role:
  - `FeedbackSample`
  - `FeedbackBuffer`
  - `load_feedback_jsonl`
- Upstream dependencies: `encode.py`, `python-chess`, JSONL schema.
- Downstream consumers: `train.py` (feedback-augmented training step).
- Operational notes: Rejects malformed or illegal rows and normalizes confidence/weights.

## 19. `generate_feedback_candidates.py`
- Category: Training pipeline
- Purpose: Bulk-extracts candidate bad-move rows from PGN corpora for faster manual labeling.
- Key classes/functions or artifact role:
  - `iter_pgn_paths`
  - `read_games_from_pgn`
  - `main`
- Upstream dependencies: PGN files and `python-chess`.
- Downstream consumers: Human labeling pipeline -> `feedback.jsonl`.
- Operational notes: Supports side filters, ply bounds, and legal-move hint truncation.

## 20. `generate_puzzles.py`
- Category: Training pipeline
- Purpose: Generates synthetic puzzle CSV rows (capture-focused tactical positions).
- Key classes/functions or artifact role:
  - `make_random_board`
  - `best_capture`
  - `generate_rows`
  - `main`
- Upstream dependencies: `python-chess`, random sampling.
- Downstream consumers: `train_puzzles.py` CSV mode or cache building.
- Operational notes: Useful for warm-up data; not a full substitute for curated real puzzle datasets.

## 21. `mark_bad_move.py`
- Category: Training pipeline
- Purpose: CLI tool to append one verified feedback row from PGN ply context.
- Key classes/functions or artifact role: `main`.
- Upstream dependencies: PGN file and legal move validation through `python-chess`.
- Downstream consumers: `feedback.jsonl` and feedback-training path.
- Operational notes: Guarantees bad/good move differ and preferred move is legal.

## 22. `mcts.py`
- Category: Core ML
- Purpose: Monte Carlo Tree Search implementation with policy/value integration and history propagation.
- Key classes/functions or artifact role:
  - `Node`
  - `mcts_run`
  - `mcts_policy_and_action`
  - `root_pi_from_visits`
- Upstream dependencies: `encode.py`, NumPy, `python-chess`, torch inference.
- Downstream consumers: `selfplay.py`, `eval.py`, GUI play scripts.
- Operational notes: Root-noise defaults are appropriate for self-play but risky in eval/inference paths if not overridden.

## 23. `model_vs_model.py`
- Category: Interface/UI
- Purpose: GUI for checkpoint-vs-checkpoint gameplay, playback navigation, PGN export, and bad-move annotation.
- Key classes/functions or artifact role:
  - `ModelVsModelApp`
  - `discover_checkpoints`
  - `save_pgn_from_moves`
- Upstream dependencies: `mcts.py`, `net.py`, `chess_board_base.py`.
- Downstream consumers: Human evaluation, data labeling, model comparison workflows.
- Operational notes: Inference settings should avoid self-play exploration noise.

## 24. `net.py`
- Category: Core ML
- Purpose: Dual-head residual neural network for policy logits and scalar value prediction.
- Key classes/functions or artifact role:
  - `ResidualBlock`
  - `AlphaZeroNet`
- Upstream dependencies: PyTorch and encoding channel/action dimensions.
- Downstream consumers: MCTS, training scripts, evaluation and GUI inference.
- Operational notes: Checkpoint compatibility depends on `channels`, `num_blocks`, `in_channels`.

## 25. `pgn_viewer.py`
- Category: Interface/UI
- Purpose: Local PGN viewer with board navigation, capture panels, and feedback-row creation.
- Key classes/functions or artifact role: `PGNViewer`.
- Upstream dependencies: `chess_board_base.py`, `python-chess`, Tkinter.
- Downstream consumers: Manual analysis and feedback labeling (`feedback.jsonl`).
- Operational notes: Includes startup path/load-latest options and side-panel responsive layout.

## 26. `play_vs_model.py`
- Category: Interface/UI
- Purpose: Human-vs-model GUI game loop with optional replay-shard/PGN export.
- Key classes/functions or artifact role:
  - `PlayVsModel`
  - `load_model`
  - `save_human_shard`
- Upstream dependencies: `mcts.py`, `net.py`, `encode.py`, `chess_board_base.py`.
- Downstream consumers: Human gameplay evaluation and optional supervised data capture.
- Operational notes: Uses MCTS for model moves and stores trajectory for optional training reuse.

## 27. `puzzle_train_data.py`
- Category: Training pipeline
- Purpose: Puzzle training data loaders, shard validators, and train-one-epoch routines.
- Key classes/functions or artifact role:
  - `PuzzleDataset`, `CachedPuzzleDataset`
  - `load_cached_shard`, `filter_valid_shards`
  - `train_one_epoch`, `train_one_epoch_from_shards`
- Upstream dependencies: `puzzles.py`, `net.py`, NumPy, torch DataLoader.
- Downstream consumers: `train_puzzles.py`, `puzzle_train_eval.py`.
- Operational notes: Includes illegal-target row filtering for corrupt/stale shard tolerance.

## 28. `puzzle_train_eval.py`
- Category: Training pipeline
- Purpose: Validation metrics and best-solved-puzzle PGN export.
- Key classes/functions or artifact role:
  - `evaluate_puzzle_validation`
  - `evaluate_puzzle_validation_from_shards`
  - `save_best_validation_pgns*`
- Upstream dependencies: `puzzle_train_data.py`, `python-chess`, torch.
- Downstream consumers: `train_puzzles.py`.
- Operational notes: Tracks top-1/top-5 and legality metrics for policy-quality diagnostics.

## 29. `puzzles.py`
- Category: Training pipeline
- Purpose: Puzzle CSV parsing, validation, canonical keying, and leakage-resistant split helpers.
- Key classes/functions or artifact role:
  - `PuzzleExample`
  - `iter_puzzle_examples`
  - `split_train_val`
- Upstream dependencies: CSV input, `encode.py`, `python-chess`.
- Downstream consumers: `train_puzzles.py` CSV mode.
- Operational notes: Drops illegal target moves and malformed rows early.

## 30. `replay_store.py`
- Category: Training pipeline
- Purpose: Persist and reload self-play replay shards (`states`, `pis`, `vs`).
- Key classes/functions or artifact role:
  - `save_shard`
  - `iter_shard_paths`
  - `load_shards_into_buffer`
- Upstream dependencies: NumPy NPZ format and replay tuple schema.
- Downstream consumers: `train.py` startup replay bootstrap.
- Operational notes: Supports newest-first loading with max sample cap.

## 31. `requirements.txt`
- Category: Config/dependency
- Purpose: Minimal dependency list for runtime and training.
- Key classes/functions or artifact role: Dependency manifest (`python-chess`, `torch`, `numpy`, `tensorboard`).
- Upstream dependencies: Python environment manager.
- Downstream consumers: Environment setup and reproducibility.
- Operational notes: Pinning strategy should be considered for stricter reproducibility.

## 32. `selfplay.py`
- Category: Training pipeline
- Purpose: Self-play game generation, trajectory construction, outcome shaping, and PGN export.
- Key classes/functions or artifact role:
  - `play_self_game`
  - `save_pgn`
  - shaping and anti-loop helpers
- Upstream dependencies: `mcts.py`, `env.py`, `encode.py`.
- Downstream consumers: `train.py` replay generation.
- Operational notes: Core producer of `(state, pi, v)` samples for RL loop.

## 33. `selfplay_train_core.py`
- Category: Training pipeline
- Purpose: Replay buffer abstraction and gradient-step functions (with optional feedback ranking loss).
- Key classes/functions or artifact role:
  - `ReplayBuffer`
  - `train_step`
  - `train_step_with_feedback`
- Upstream dependencies: NumPy, torch, augmentation helpers from `encode.py`.
- Downstream consumers: `train.py`.
- Operational notes: Applies gradient clipping and optional color-flip augmentation on sampled batches.

## 34. `tests.py`
- Category: Tests
- Purpose: Unit-level checks for encoding invariants, augmentation behavior, and MCTS backup sign/history behavior.
- Key classes/functions or artifact role:
  - `test_move_index_unique_in_startpos`
  - `test_board_tensor_history`
  - `test_mcts_backup_sign`
  - `test_mcts_with_history`
- Upstream dependencies: Core modules under test.
- Downstream consumers: Developer regression validation.
- Operational notes: Current history test logic has a stale-move issue causing false failures.

## 35. `train.py`
- Category: Training pipeline
- Purpose: Main self-play RL orchestration loop: self-play generation, replay training, gating, checkpointing, eval logging.
- Key classes/functions or artifact role:
  - `_cosine_lr`
  - checkpoint loading path
  - iteration loop and gate logic
- Upstream dependencies: `selfplay.py`, `selfplay_train_core.py`, `eval.py`, feedback loader.
- Downstream consumers: Produces `checkpoint_latest.pt`, replay shards, TensorBoard logs.
- Operational notes: Puzzle-init checkpoint policy and optimizer-state loading behavior materially affect RL startup stability.

## 36. `train_puzzles.py`
- Category: Training pipeline
- Purpose: Puzzle-supervised training entry point (CSV mode and cache-shard mode), with auto-tune and checkpoint export.
- Key classes/functions or artifact role:
  - `_benchmark_cache_mode`
  - `_benchmark_csv_mode`
  - `main`
- Upstream dependencies: `puzzles.py`, `puzzle_train_data.py`, `puzzle_train_eval.py`, `net.py`.
- Downstream consumers: Puzzle checkpoints and validation PGN exports.
- Operational notes: Supports both exploratory overfit-debug runs and larger cached training loops.
