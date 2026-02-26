from __future__ import annotations

import argparse
import os
import time
import numpy as np
import tkinter as tk
from tkinter import messagebox

import chess
import chess.pgn
import torch

from net import AlphaZeroNet
from mcts import Node, mcts_policy_and_action, reuse_root_after_action
from encode import (
    board_to_tensor, move_to_index, action_to_move, ACTION_SIZE, IN_CHANNELS,
)
from chess_board_base import (
    BoardRenderer, create_capture_grid, recompute_captures, update_capture_display,
)

SAVE_TRAINING_SAMPLES = True
HUMAN_REPLAY_DIR = "replay_human"
HUMAN_PGN_DIR = "human_games"
INFER_DIRICHLET_ALPHA = 0.0
INFER_DIRICHLET_EPS = 0.0


def save_pgn_from_moves(moves, result_str, out_dir, human_is_white):
    os.makedirs(out_dir, exist_ok=True)
    game = chess.pgn.Game()
    game.headers["Event"] = "HumanVsNet"
    game.headers["Site"] = "Local"
    game.headers["Date"] = time.strftime("%Y.%m.%d")
    game.headers["White"] = "Human" if human_is_white else "Net"
    game.headers["Black"] = "Net" if human_is_white else "Human"
    game.headers["Result"] = result_str
    node = game
    board = chess.Board()
    for mv in moves:
        node = node.add_variation(mv)
        board.push(mv)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"human_vs_net_{ts}_{np.random.randint(100000):05d}.pgn")
    with open(path, "w", encoding="utf-8") as f:
        print(game, file=f, end="\n\n")
    return path


def save_human_shard(samples, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    states = np.stack([s for (s, pi, v) in samples], axis=0).astype(np.float32)
    pis = np.stack([pi for (s, pi, v) in samples], axis=0).astype(np.float32)
    vs = np.array([v for (s, pi, v) in samples], dtype=np.float32)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"human_shard_{ts}_{np.random.randint(100000):05d}.npz")
    np.savez_compressed(path, states=states, pis=pis, vs=vs)
    return path


class PlayVsModel(tk.Tk):
    def __init__(
        self, net, device="cpu", num_sims=50,
        save_training_samples=SAVE_TRAINING_SAMPLES,
        human_replay_dir=HUMAN_REPLAY_DIR,
        human_pgn_dir=HUMAN_PGN_DIR,
        start_as_black=False,
    ):
        super().__init__()
        self.title("Play vs Model")
        self.resizable(True, True)
        self.minsize(900, 560)

        self.net = net
        self.device = device
        self.num_sims = num_sims
        self.save_training_samples = bool(save_training_samples)
        self.human_replay_dir = human_replay_dir
        self.human_pgn_dir = human_pgn_dir

        self.margin = 10
        self._resize_after_id: str | None = None

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1, minsize=180)
        self.grid_columnconfigure(1, weight=4, minsize=560)
        self.grid_columnconfigure(2, weight=1, minsize=180)

        self.left_frame = tk.Frame(self)
        self.left_frame.grid(row=0, column=0, padx=(10, 6), pady=10, sticky="nsew")
        self.center_frame = tk.Frame(self)
        self.center_frame.grid(row=0, column=1, padx=6, pady=10, sticky="nsew")
        self.right_frame = tk.Frame(self)
        self.right_frame.grid(row=0, column=2, padx=(6, 10), pady=10, sticky="nsew")

        self.center_frame.grid_rowconfigure(0, weight=1)
        for col in range(8):
            self.center_frame.grid_columnconfigure(col, weight=1)

        tk.Label(self.left_frame, text="White captured", font=("Arial", 12, "bold")).pack(anchor="n")
        self.white_capture_slots = create_capture_grid(self.left_frame)
        tk.Label(self.right_frame, text="Black captured", font=("Arial", 12, "bold")).pack(anchor="n")
        self.black_capture_slots = create_capture_grid(self.right_frame)

        self.canvas = tk.Canvas(self.center_frame, width=530, height=530)
        self.canvas.grid(row=0, column=0, columnspan=8, sticky="nsew")
        self.canvas.bind("<Configure>", self.on_canvas_resized)
        self.canvas.bind("<Button-1>", self.on_click)

        self.renderer = BoardRenderer(self.canvas)

        self.btn_new = tk.Button(self.center_frame, text="New Game", command=self.new_game)
        self.btn_new.grid(row=1, column=0, sticky="ew", padx=4, pady=(8, 0))
        self.btn_undo = tk.Button(self.center_frame, text="Undo (2 plies)", command=self.undo_two)
        self.btn_undo.grid(row=1, column=1, sticky="ew", padx=4, pady=(8, 0))
        self.btn_flip = tk.Button(self.center_frame, text="Flip Board", command=self.flip_board)
        self.btn_flip.grid(row=1, column=2, sticky="ew", padx=4, pady=(8, 0))

        self.play_as_black_var = tk.BooleanVar(value=bool(start_as_black))
        self.chk_black = tk.Checkbutton(self.center_frame, text="Play as Black",
                                        variable=self.play_as_black_var, command=self.on_toggle_side)
        self.chk_black.grid(row=1, column=3, sticky="w", padx=4, pady=(8, 0))

        self.btn_prev = tk.Button(self.center_frame, text="◀ Prev", command=self.prev_move, state="disabled")
        self.btn_prev.grid(row=1, column=4, sticky="ew", padx=4, pady=(8, 0))
        self.btn_next = tk.Button(self.center_frame, text="Next ▶", command=self.next_move, state="disabled")
        self.btn_next.grid(row=1, column=5, sticky="ew", padx=4, pady=(8, 0))
        self.btn_latest = tk.Button(self.center_frame, text="Latest", command=self.go_latest, state="disabled")
        self.btn_latest.grid(row=1, column=6, sticky="ew", padx=4, pady=(8, 0))

        self.lbl = tk.Label(self.center_frame, text="", anchor="w")
        self.lbl.grid(row=2, column=0, columnspan=8, sticky="ew", pady=(8, 0))

        self.board = chess.Board()
        self.view_board = chess.Board()
        self.moves: list[chess.Move] = []
        self.selected_sq: int | None = None
        self.flipped = False
        self.sans: list[str] = []
        self.view_ply = 0
        self.traj: list = []
        self.board_history: list[chess.Board] = []
        self.root = Node(self.board.copy(stack=False))

        self.bind("<Left>", lambda e: self.prev_move())
        self.bind("<Right>", lambda e: self.next_move())

        self.update_label()
        self.update_nav_buttons()
        self._update_side_panels()
        self.draw()
        if not self.human_is_white:
            self.after(50, self.net_move)

    @property
    def human_is_white(self) -> bool:
        return not self.play_as_black_var.get()

    def on_toggle_side(self):
        self.update_label()

    def update_label(self, extra=""):
        side = "White" if self.human_is_white else "Black"
        if self.view_ply == 0:
            move_info = "Start position"
        else:
            san = self.sans[self.view_ply - 1] if self.view_ply - 1 < len(self.sans) else self.moves[self.view_ply - 1].uci()
            move_no = (self.view_ply + 1) // 2
            turn_side = "White" if self.view_ply % 2 == 1 else "Black"
            move_info = f"Move {move_no} ({turn_side}): {san}"
        msg = f"You are {side}. {move_info}. Click piece then destination."
        if self.view_ply != len(self.moves) and not self.board.is_game_over(claim_draw=True):
            msg += " (Inspect mode: go to Latest to play.)"
        if extra:
            msg += " " + extra
        self.lbl.config(text=msg)

    def new_game(self):
        self.board = chess.Board()
        self.view_board = self.board.copy(stack=False)
        self.moves = []
        self.sans = []
        self.selected_sq = None
        self.view_ply = 0
        self.traj = []
        self.board_history = []
        self.root = Node(self.board.copy(stack=False))
        self.update_label("New game.")
        self.update_nav_buttons()
        self._update_side_panels()
        self.draw()
        if not self.human_is_white:
            self.after(50, self.net_move)

    def flip_board(self):
        self.flipped = not self.flipped
        self.renderer.flipped = self.flipped
        self.draw()

    def undo_two(self):
        if len(self.moves) >= 2:
            self.board.pop()
            self.board.pop()
            self.moves = self.moves[:-2]
            self.sans = self.sans[:-2]
            if len(self.traj) >= 2:
                self.traj = self.traj[:-2]
            if len(self.board_history) >= 2:
                self.board_history = self.board_history[:-2]
            self.root = Node(self.board.copy(stack=False))
            self.selected_sq = None
            self.set_view_ply(len(self.moves))
            self.update_label("Undid last two plies.")

    # ---- Input ----
    def on_click(self, event):
        if self.board.is_game_over(claim_draw=True):
            return
        if self.view_ply != len(self.moves):
            self.update_label("You are viewing an earlier move. Press Latest to continue.")
            return
        human_color = chess.WHITE if self.human_is_white else chess.BLACK
        if self.board.turn != human_color:
            return
        sq = self.renderer.pixel_to_square(event.x, event.y)
        if sq is None:
            return
        if self.selected_sq is None:
            piece = self.board.piece_at(sq)
            if piece is None or piece.color != human_color:
                return
            self.selected_sq = sq
            self.draw()
            return
        from_sq = self.selected_sq
        to_sq = sq
        self.selected_sq = None
        move = chess.Move(from_sq, to_sq)
        if self.is_pawn_promotion(move):
            promo = self.ask_promotion_piece()
            if promo is None:
                self.draw()
                return
            move = chess.Move(from_sq, to_sq, promotion=promo)
        if move not in self.board.legal_moves:
            self.update_label("Illegal move.")
            self.draw()
            return
        self.apply_human_move(move)
        if self.board.is_game_over(claim_draw=True):
            self.finish_game()
            return
        self.after(10, self.net_move)

    def apply_human_move(self, move):
        state = board_to_tensor(self.board, history=self.board_history)
        pi = np.zeros((ACTION_SIZE,), dtype=np.float32)
        pi[move_to_index(move)] = 1.0
        self.traj.append((state, pi, self.board.turn))
        self.sans.append(self.board.san(move))
        self.board_history.append(self.board.copy(stack=False))
        self.board.push(move)
        self.moves.append(move)
        self.root = Node(self.board.copy(stack=False))
        self.set_view_ply(len(self.moves))

    # ---- Net ----
    def net_move(self):
        if self.board.is_game_over(claim_draw=True):
            return
        net_color = chess.BLACK if self.human_is_white else chess.WHITE
        if self.board.turn != net_color:
            return
        with torch.inference_mode():
            pi, action = mcts_policy_and_action(
                self.net, root=self.root, num_sims=self.num_sims,
                temperature=1e-6, device=self.device, history=self.board_history,
                dirichlet_alpha=INFER_DIRICHLET_ALPHA,
                dirichlet_eps=INFER_DIRICHLET_EPS,
            )
        state = board_to_tensor(self.board, history=self.board_history)
        self.traj.append((state, pi.astype(np.float32), self.board.turn))
        mv = action_to_move(action)
        if mv not in self.board.legal_moves:
            mv = np.random.choice(list(self.board.legal_moves))
        self.sans.append(self.board.san(mv))
        self.board_history.append(self.board.copy(stack=False))
        self.board.push(mv)
        self.moves.append(mv)
        self.root = reuse_root_after_action(self.root, action)
        self.root.board = self.board.copy(stack=False)
        if self.board.is_game_over(claim_draw=True):
            self.finish_game()
            return
        self.set_view_ply(len(self.moves))

    # ---- Finish ----
    def finish_game(self):
        res = self.board.result(claim_draw=True)
        z_white = 1.0 if res == "1-0" else (-1.0 if res == "0-1" else 0.0)
        pgn_path = save_pgn_from_moves(self.moves, res, self.human_pgn_dir, human_is_white=self.human_is_white)
        samples = [(s, pi.astype(np.float32), float(z_white if tp == chess.WHITE else -z_white))
                    for s, pi, tp in self.traj]
        shard_path = None
        if self.save_training_samples and samples:
            shard_path = save_human_shard(samples, self.human_replay_dir)
        msg = f"Game over: {res}\nSaved PGN: {pgn_path}"
        if shard_path:
            msg += f"\nSaved training shard: {shard_path}"
        self.lbl.config(text=msg)
        self.set_view_ply(len(self.moves))
        self.draw()
        messagebox.showinfo("Game finished", msg)

    # ---- Helpers ----
    def ask_promotion_piece(self):
        win = tk.Toplevel(self)
        win.title("Promotion")
        win.resizable(False, False)
        choice = {"p": None}
        def pick(pt):
            choice["p"] = pt
            win.destroy()
        tk.Label(win, text="Promote to:").pack(padx=10, pady=10)
        row = tk.Frame(win)
        row.pack(padx=10, pady=(0, 10))
        tk.Button(row, text="Queen", width=8, command=lambda: pick(chess.QUEEN)).pack(side="left", padx=4)
        tk.Button(row, text="Rook", width=8, command=lambda: pick(chess.ROOK)).pack(side="left", padx=4)
        tk.Button(row, text="Bishop", width=8, command=lambda: pick(chess.BISHOP)).pack(side="left", padx=4)
        tk.Button(row, text="Knight", width=8, command=lambda: pick(chess.KNIGHT)).pack(side="left", padx=4)
        win.transient(self)
        win.grab_set()
        self.wait_window(win)
        return choice["p"]

    def is_pawn_promotion(self, mv):
        piece = self.board.piece_at(mv.from_square)
        if piece is None or piece.piece_type != chess.PAWN:
            return False
        rank_to = chess.square_rank(mv.to_square)
        return (piece.color == chess.WHITE and rank_to == 7) or (piece.color == chess.BLACK and rank_to == 0)

    def _update_side_panels(self):
        wcap, bcap = recompute_captures(self.moves, self.view_ply)
        update_capture_display(self.white_capture_slots, wcap)
        update_capture_display(self.black_capture_slots, bcap)

    def update_nav_buttons(self):
        self.btn_prev.config(state="normal" if self.view_ply > 0 else "disabled")
        self.btn_next.config(state="normal" if self.view_ply < len(self.moves) else "disabled")
        self.btn_latest.config(state="normal" if self.view_ply < len(self.moves) else "disabled")

    def set_view_ply(self, target_ply):
        target = max(0, min(target_ply, len(self.moves)))
        b = chess.Board()
        for i in range(target):
            b.push(self.moves[i])
        self.view_board = b
        self.view_ply = target
        if self.view_ply != len(self.moves):
            self.selected_sq = None
        self.update_nav_buttons()
        self._update_side_panels()
        self.update_label()
        self.draw()

    def prev_move(self):
        self.set_view_ply(self.view_ply - 1)

    def next_move(self):
        self.set_view_ply(self.view_ply + 1)

    def go_latest(self):
        self.set_view_ply(len(self.moves))

    def on_canvas_resized(self, _event=None):
        if self._resize_after_id is not None:
            self.after_cancel(self._resize_after_id)
        self._resize_after_id = self.after(30, self.draw)

    def draw(self):
        self._resize_after_id = None
        last_move = self.moves[self.view_ply - 1] if self.view_ply > 0 else None
        selected = self.selected_sq if self.view_ply == len(self.moves) else None
        self.renderer.draw(self.view_board, last_move=last_move, selected_sq=selected)
        self.update_idletasks()


def load_model(device="cpu", checkpoint_path="checkpoint_latest.pt",
               channels=128, num_blocks=10):
    net = AlphaZeroNet(in_channels=IN_CHANNELS, channels=channels,
                       num_blocks=num_blocks).to(device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(payload, dict) and "model_state_dict" in payload:
            net.load_state_dict(payload["model_state_dict"])
            ch = payload.get("channels", channels)
            nb = payload.get("num_blocks", num_blocks)
            if ch != channels or nb != num_blocks:
                net = AlphaZeroNet(in_channels=IN_CHANNELS, channels=ch, num_blocks=nb).to(device)
                net.load_state_dict(payload["model_state_dict"])
            it = payload.get("iter")
            tag = f" (iter={it})" if it is not None else ""
            print(f"Loaded {checkpoint_path}{tag}")
        else:
            net.load_state_dict(payload)
            print(f"Loaded {checkpoint_path}")
    else:
        print(f"No checkpoint found at '{checkpoint_path}', using untrained model.")
    net.eval()
    return net


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play a local GUI game vs model.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_latest.pt")
    parser.add_argument("--num_sims", type=int, default=50)
    parser.add_argument("--play_as_black", action="store_true")
    parser.add_argument("--save_training_samples", action="store_true")
    parser.add_argument("--human_replay_dir", type=str, default=HUMAN_REPLAY_DIR)
    parser.add_argument("--human_pgn_dir", type=str, default=HUMAN_PGN_DIR)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--num_blocks", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.human_pgn_dir, exist_ok=True)
    os.makedirs(args.human_replay_dir, exist_ok=True)

    net = load_model(device=args.device, checkpoint_path=args.checkpoint,
                     channels=args.channels, num_blocks=args.num_blocks)
    app = PlayVsModel(
        net, device=args.device, num_sims=args.num_sims,
        save_training_samples=args.save_training_samples,
        human_replay_dir=args.human_replay_dir,
        human_pgn_dir=args.human_pgn_dir,
        start_as_black=args.play_as_black,
    )
    app.mainloop()
