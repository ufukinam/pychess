# Self-Play Training Roadmap

This roadmap gives a concrete path to improve model strength while reducing regressions.

## 1. Ground Rules

1. Keep two self-play checkpoints:
   - `checkpoint_latest.pt`: latest accepted model.
   - `checkpoint_best.pt`: best model by configured score metric.
2. Track every iteration in `training_scoreboard.jsonl`.
3. Use venv Python for all commands:
   - `.\.venv\Scripts\python.exe ...`

## 2. Scoring Policy

1. Gate acceptance (`checkpoint_latest.pt`) uses:
   - `gate_score_mode=score` (default, compatible with old threshold tuning), or
   - `gate_score_mode=ci_low` (stricter, lower-variance acceptance).
2. Best checkpoint promotion (`checkpoint_best.pt`) uses random-baseline eval:
   - `best_score_mode=score` for speed,
   - `best_score_mode=ci_low` for safer promotion.
3. Promotion rule:
   - `best_promotion_rule=and_gate_eval` to require both gate and eval improvements.
   - `best_promotion_rule=eval_only` to promote only by eval metric.
4. Recommended production default:
   - `gate_score_mode=score`
   - `best_score_mode=ci_low`
   - `best_promotion_rule=and_gate_eval`

## 3. Stage Plan

### Stage A: Puzzle Baseline

Goal: produce a stable transfer model.

1. Train puzzle model to convergence trend (top1 plateaus).
2. Start self-play from `checkpoint_puzzle_best.pt`, not latest puzzle.

### Stage B: Calibration Run (Short)

Goal: validate settings before long runs.

1. Run 3-5 iterations with low game count.
2. Confirm:
   - gate accepts at least some iterations,
   - eval score is not collapsing,
   - scoreboard JSONL is populated.
3. If acceptance is near 0%, reduce strictness:
   - lower `gate_min_score`, or
   - switch `gate_score_mode` to `score`.

### Stage C: Production Run (Long)

Goal: maximize `checkpoint_best.pt` score metric.

1. Increase self-play samples (`games_per_iter`) before increasing LR.
2. Keep eval cadence regular (`eval_every=1` or `2`).
3. Prefer larger `eval_games` for lower variance when promoting best.

### Stage D: Controlled Ablations

Goal: isolate what actually helps.

Run one change at a time:

1. Search budget: `num_sims`, `eval_num_sims`.
2. Replay quality: `replay_maxlen`, augmentation on/off.
3. Optimization: `lr`, `train_batches`, `batch_size`.
4. Draw/loop handling: no-progress and repeat penalties.

## 4. Suggested Commands

### 4.1 Calibration

```powershell
.\.venv\Scripts\python.exe train.py `
  --prefer_puzzle_init `
  --puzzle_checkpoint checkpoint_puzzle_best.pt `
  --iters 4 `
  --games_per_iter 8 `
  --train_batches 8 `
  --batch_size 64 `
  --num_sims 100 `
  --eval_num_sims 40 `
  --gate_games 20 `
  --gate_min_score 0.52 `
  --gate_score_mode score `
  --eval_every 1 `
  --eval_games 16 `
  --best_score_mode ci_low `
  --best_promotion_rule and_gate_eval `
  --latest_checkpoint checkpoint_latest.pt `
  --best_checkpoint checkpoint_best.pt `
  --scoreboard_jsonl training_scoreboard.jsonl
```

### 4.2 Production

```powershell
.\.venv\Scripts\python.exe train.py `
  --init_checkpoint checkpoint_latest.pt `
  --iters 20 `
  --games_per_iter 40 `
  --train_batches 32 `
  --batch_size 64 `
  --num_sims 200 `
  --eval_num_sims 60 `
  --gate_games 30 `
  --gate_min_score 0.52 `
  --gate_score_mode score `
  --eval_every 1 `
  --eval_games 24 `
  --best_score_mode ci_low `
  --best_promotion_rule and_gate_eval `
  --latest_checkpoint checkpoint_latest.pt `
  --best_checkpoint checkpoint_best.pt `
  --scoreboard_jsonl training_scoreboard.jsonl
```

## 5. Weekly Review Checklist

1. `checkpoint_best.pt` metric trend is non-decreasing.
2. Acceptance ratio is not near 0% or 100% for long stretches.
3. Eval CI width is shrinking or stable (enough eval games).
4. Draw rate is not dominating due loop-heavy policies.
5. Parameters changed this week are logged with run notes.

If these checks fail, pause long runs and do a short calibration run before continuing.
