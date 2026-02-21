# PyChess Workbook (Step-by-Step Execution)

Use this workbook while following `docs/TEACHING_PLAN.md`.

Mark each item when complete.

## Module 0 Checklist
- [X] Ran CLI help and all subcommand help.
- [X] Opened GUI and identified each tab purpose.
- [X] Mapped tab -> subcommand -> script.

## Module 1 Checklist
- [X] Verified `ACTION_SIZE = 20480`.
- [X] Tested move <-> index conversion.
- [X] Verified legal moves are true in legal mask.
- [X] Printed one board tensor and explained planes.

## Module 2 Checklist
- [X] Ran one forward pass and checked output shapes.
- [X] Explained policy head and value head roles.

## Module 3 Checklist
- [ ] Traced one MCTS action path through select/expand/backup.
- [ ] Confirmed selected action is legal.

## Module 4 Checklist
- [ ] Ran short self-play.
- [ ] Opened generated PGN.
- [ ] Explained sample tuple `(state, pi, v)` creation.

## Module 5 Checklist
- [ ] Ran tiny self-play training loop (`iters` small).
- [ ] Observed replay save/load behavior.
- [ ] Located and explained checkpoint files.

## Module 6 Checklist
- [ ] Ran puzzle train from CSV (small limit).
- [ ] Ran puzzle train from cache.
- [ ] Compared `val_top1`, `val_top5`, legality metrics.

## Module 7 Checklist
- [ ] Built small cache with `compression=none`.
- [ ] Built larger cache with tuned workers.
- [ ] Inspected `manifest.json`.
- [ ] Verified shard schema keys.

## Module 8 Checklist
- [ ] Collected `[TrainTiming]` and `[EpochTiming]`.
- [ ] Ran `--auto_tune_cpu --tune_only`.
- [ ] Selected best config using `samples/s`.
- [ ] Re-ran training with selected config.

## Module 9 Checklist
- [ ] Executed same training config via CLI and GUI.
- [ ] Confirmed effective arguments match.
- [ ] Verified logs match expected behavior.

## Module 10 Checklist
- [ ] Ran `python tests.py`.
- [ ] Ran overfit sanity mode.
- [ ] Wrote short regression note (what changed, what remained stable).

---

## Milestone Acceptance Gates

### Gate A: Core Engine
- [ ] Encoding + net + MCTS understood and traceable.

### Gate B: RL Loop
- [ ] Self-play -> replay -> train loop runs end-to-end.

### Gate C: Puzzle Pipeline
- [ ] CSV and cache modes both validated.

### Gate D: Performance
- [ ] Throughput measured and tuned with evidence.

### Gate E: Operations
- [ ] CLI/GUI parity verified.

### Gate F: Reliability
- [ ] Tests and sanity checks pass.

---

## Reflection Prompts (write answers)
1. Why is legality masking required in both training and evaluation?
2. Why can higher CPU usage fail to improve throughput?
3. When should you use puzzle pretraining before self-play?
4. Which cache settings are best for fast iteration vs low disk?
5. If `val_top1` rises but gameplay weakens, what would you inspect first?

