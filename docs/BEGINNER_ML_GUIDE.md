# Beginner Guide: How This Chess ML Project Works

This project combines three main ideas:
- Chess rules and board state management (`python-chess`)
- Neural network prediction (`PyTorch`)
- Search on top of predictions (MCTS in `mcts.py`)

## 1) Project Flow (Simple Mental Model)

1. Convert a chess board into numbers (`encode.py`).
2. Neural net predicts:
- policy: which move looks good
- value: who is winning
3. MCTS refines policy by running many lookahead simulations (`mcts.py`).
4. Train from:
- self-play games (`train.py`, `selfplay.py`)
- puzzle supervision (`train_puzzles.py`, `puzzles.py`)

## 2) Libraries Used and Why

### `python-chess`
- Purpose: reliable chess rules, legal moves, FEN/PGN parsing.
- Key classes/functions used:
- `chess.Board()`: full game state object.
- `board.legal_moves`: legal move generator.
- `board.push(move)`, `board.pop()`: apply/undo moves.
- `board.result()`, `board.is_game_over()`: terminal checks.
- `chess.Move.from_uci("e2e4")`: parse move text.
- `chess.pgn.Game()`: build/save PGN games.

Why used:
- Writing chess rules manually is error-prone; this library is battle-tested.

### `numpy` (`np`)
- Purpose: fast array math and storage for training data.
- Key functions used:
- `np.zeros`, `np.stack`, `np.array`: build tensors/arrays.
- `np.random.choice`, `np.random.permutation`: sampling/shuffling.
- `np.packbits` / `np.unpackbits`: compress/decompress legal masks.
- `np.savez_compressed`, `np.load`: shard persistence.

Why used:
- Efficient CPU-side data processing before tensors go into PyTorch.

### `torch` / `torch.nn` / `torch.nn.functional`
- Purpose: define/train neural network.
- Key APIs used:
- `nn.Conv2d`, `nn.Linear`, `nn.BatchNorm2d`: model layers.
- `F.relu`, `torch.tanh`: activations.
- `F.cross_entropy`: policy classification loss.
- `F.mse_loss`: value regression loss.
- `torch.optim.Adam`: optimizer.
- `@torch.inference_mode()`: fast eval without gradients.
- `tensor.to(device)`: move data to CPU/GPU.

Why used:
- Standard deep-learning framework with strong ecosystem and GPU support.

### `torch.utils.data`
- `Dataset`, `DataLoader` provide minibatch iteration and shuffling.

Why used:
- Cleaner and safer batch pipeline for training loops.

### `tkinter`
- Purpose: local GUI tools (`play_vs_model.py`, `pgn_viewer.py`, `chess_gui.py`).
- Used for buttons/canvas/events to visualize and control games.

## 3) Important File Roles

- `encode.py`: board/move encoding for model/search.
- `net.py`: neural network architecture.
- `mcts.py`: search algorithm that improves raw model policy.
- `selfplay.py`: self-play game generation and training sample creation.
- `selfplay_train_core.py`: replay buffer + single train step.
- `train.py`: full self-play training loop.
- `puzzles.py`: puzzle CSV parsing and split logic.
- `puzzle_train_data.py`: puzzle datasets and epoch training helpers.
- `puzzle_train_eval.py`: puzzle validation metrics and PGN export.
- `train_puzzles.py`: puzzle training entry point.

## 4) Why We Mask Illegal Moves

The policy head always outputs a fixed vector (`ACTION_SIZE = 20480`), but only a few actions are legal in a position.
So the code builds a `legal_mask` and sets illegal logits to `-1e9` before softmax/loss.
Without this, the model can learn to assign probability to impossible moves.

## 5) Why MCTS Flips Value Sign in Backup

Value at a node is from the side-to-move perspective.
After one move, perspective switches to the opponent.
So backup uses `v = -v` each ply to keep values consistent along the path.

## 6) How to Read This Code as a Beginner

Recommended order:
1. `encode.py`
2. `net.py`
3. `mcts.py`
4. `selfplay.py`
5. `train.py`
6. `puzzles.py` + `train_puzzles.py`

Read one file at a time and track:
- inputs
- outputs
- where outputs are consumed next
