# Codebase Audit (Tracked Files Only)

Audit date: 2026-02-26
Repository root: `c:\Users\ZB_UFUK\pychess`
Scope lock: `git ls-files` only

## 1. Scope and Method

- Authoritative inventory source: `git ls-files`
- Tracked file count at audit time: `36`
- Existing docs were left unchanged:
  - `docs/BEGINNER_ML_GUIDE.md`
  - `docs/TEACHING_PLAN.md`
  - `docs/WORKBOOK.md`
- This audit covers integrity signals that materially affect training quality, evaluation stability, and future maintainability.

## 2. Tracked-File Classification Snapshot

Classification categories required by the plan:

- Core ML: 4
- Training pipeline: 13
- Evaluation: 1
- Interface/UI: 6
- Data artifacts: 5
- Tests: 1
- Docs: 4
- Config/dependency: 2

Total: 36

## 3. Integrity Checks and Outcomes

### 3.1 Test runner integrity

- Non-venv invocation was intentionally checked and failed:
  - `python tests.py`
- Failure happened before tests run due environment mismatch:
  - `ImportError: DLL load failed while importing _C`
  - Interpreter path indicates system Python site-packages, not project venv.

Command:

```powershell
.\.venv\Scripts\python.exe tests.py
```

Outcome:

- Early tests passed (encoding basics), then failed in `test_board_tensor_history`.
- Failure signature:
  - `AssertionError: push() expects move to be pseudo-legal, but got g1f3 ...`
- Root cause for this failure is in test logic (see Critical Issue 3), not in model runtime.

### 3.2 Cache shard schema integrity

Inspection commands were run against:

- `puzzle_cache_lichess/train_shard_00000.npz`
- `puzzle_cache_lichess/val_shard_00000.npz`

Observed schema:

- Train shard keys: `states`, `target_idx`, `legal_masks_packed`
- Val shard keys: `states`, `target_idx`, `legal_masks_packed`, `fens`, `moves`, `puzzle_ids`

Observed shapes and dtypes (sample shard):

- `states`: `(4096, 67, 8, 8)`, `float16`
- `target_idx`: `(4096,)`, `int32`
- `legal_masks_packed`: `(4096, 2560)`, `uint8`

This matches current loader expectations in `puzzle_train_data.py`.

### 3.3 Checkpoint payload integrity

Command inspected `checkpoint_puzzle_best.pt` keys.

Observed keys:

- `model_state_dict`
- `optimizer_state_dict`
- `epoch`
- `in_channels`
- `channels`
- `num_blocks`

Implication:

- Puzzle checkpoints include optimizer state, which interacts with self-play init logic in `train.py` (see Critical Issue 2).

## 4. Environment Caveat

All verification and training commands should use the project venv to avoid interpreter/DLL mismatch:

```powershell
.\.venv\Scripts\python.exe <script>.py ...
```

Using system `python` in this workspace currently produces a PyTorch import failure and invalidates integrity checks.

## 5. Critical Issues (Patch-Ready Specs) PATCHED

Findings are ordered by severity and expected impact on model quality or decision reliability.

### 5.1 High: Eval/gameplay paths inherit self-play Dirichlet noise defaults

Evidence:

- `mcts.py:171` and `mcts.py:266` define default `dirichlet_eps=0.25`.
- Eval/inference call sites do not override this:
  - `eval.py:34`
  - `eval.py:81`
  - `play_vs_model.py:274`
  - `model_vs_model.py:238`

Observed behavior check:

- Repeated deterministic-action probe (`temperature=1e-6`) with a constant net produced:
  - `unique_actions 5` with default noise path.
  - `unique_actions 1` when forcing `dirichlet_eps=0.0`.

Root cause:

- Self-play exploration noise configuration is reused in evaluation and interactive inference, where noise should be disabled.

ML impact:

- Noisy gating/eval outcomes.
- Unstable model acceptance/rejection around threshold.
- Perceived weaker gameplay consistency in UI tools.

Patch-ready remediation:

1. In each evaluation/inference call site, explicitly pass:
   - `dirichlet_eps=0.0`
   - optionally `dirichlet_alpha=0.0` for clarity.
2. Keep self-play default exploration unchanged (`selfplay.py` path).
3. Optionally define local constants for readability:
   - `EVAL_DIRICHLET_EPS = 0.0`
   - `EVAL_DIRICHLET_ALPHA = 0.0`

Target files:

- `eval.py`
- `play_vs_model.py`
- `model_vs_model.py`

Verification commands:

```powershell
.\.venv\Scripts\python.exe -c "import chess, torch; from mcts import Node, mcts_policy_and_action; from encode import ACTION_SIZE; class Z(torch.nn.Module):`n    def forward(self,x):`n        b=x.shape[0]; return torch.zeros((b,ACTION_SIZE),dtype=x.dtype,device=x.device), torch.zeros((b,),dtype=x.dtype,device=x.device); net=Z(); b=chess.Board(); acts=[];`nfor _ in range(6):`n    r=Node(b.copy(stack=False)); _,a=mcts_policy_and_action(net,r,num_sims=40,temperature=1e-6,device='cpu',dirichlet_eps=0.0,dirichlet_alpha=0.0); acts.append(a);`nprint(len(set(acts)),acts)"
```

Expected: stable action choice count (typically 1 unique action in this probe).

### 5.2 High: Puzzle-init path loads optimizer state into RL loop

Evidence:

- `train.py:119` defines `_load_checkpoint`.
- `train.py:123-124` loads `optimizer_state_dict` whenever present.
- Puzzle init route:
  - `train.py:130-132` uses `args.puzzle_checkpoint` when `--prefer_puzzle_init`.
- Puzzle checkpoints currently contain optimizer state (integrity check result above).

Root cause:

- Same checkpoint loader behavior is used for both RL resumes and puzzle-to-RL transfer.

ML impact:

- AdamW moments/statistics from supervised puzzle training can bias early RL updates.
- Can destabilize gating and obscure true impact of newly generated replay.

Patch-ready remediation:

1. Change loader signature to allow explicit optimizer policy:
   - from: `_load_checkpoint(path: str) -> int | None`
   - to: `_load_checkpoint(path: str, load_optimizer: bool = True) -> int | None`
2. Apply policy:
   - RL resume (`--init_checkpoint`): `load_optimizer=True`
   - puzzle transfer (`--prefer_puzzle_init` path): default `load_optimizer=False`
3. Add explicit CLI override for advanced users:
   - `--load_optimizer_from_puzzle_init` (default false).
4. Log which policy was applied on startup.

Target file:

- `train.py`

Verification commands:

```powershell
.\.venv\Scripts\python.exe train.py --prefer_puzzle_init --iters 1 --games_per_iter 1 --train_batches 1 --num_sims 20 --eval_num_sims 10
```

Check startup log confirms optimizer policy for puzzle-init path.

### 5.3 Medium: Test suite false failure in history test setup

Evidence:

- `tests.py:53` creates a static move list once from initial board.
- `tests.py:56` pushes each move after board state has changed.
- This can make later moves illegal in the updated position.

Root cause:

- Test uses stale legal move candidates across sequential board states.

Impact:

- False-negative unit-test failure.
- Reduces trust in regression checks for encoding/history features.

Patch-ready remediation:

Replace static list logic with one of:

1. Dynamic legal move sampling each ply:
   - each iteration pick from current `board.legal_moves`.
2. Hard-coded legal opening sequence replayed in order.

Recommended low-risk patch:

- In `test_board_tensor_history`, build history with:
  - loop `for _ in range(4):`
  - `mv = next(iter(board.legal_moves))`
  - push and continue.

Target file:

- `tests.py`

Verification command:

```powershell
.\.venv\Scripts\python.exe tests.py
```

Expected: all tests pass.

## 6. Test Cases and Scenarios to Track

The following scenarios should be executed after issue remediation:

1. Eval determinism check
   - Confirm repeated deterministic move selection remains stable with root noise disabled.
2. Gate stability check
   - Compare gate acceptance variance before/after eval noise removal.
3. Optimizer transfer ablation
   - Compare early RL metrics with and without puzzle optimizer-state restore.
4. Regression test check
   - Full `.venv` test pass after history test fix.
5. Catalog completeness check
   - Tracked-file count equals file-catalog entry count.

## 7. Documentation Quality Gate Results (This Pass)

- New docs created as separate files: yes.
- Existing docs modified: no.
- File-catalog target scope (`git ls-files`): yes.
- Runtime code/API changed in this pass: no.
