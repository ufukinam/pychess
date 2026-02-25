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
from encode import board_to_tensor, move_to_index, ACTION_SIZE


UNICODE_PIECE = {
    "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
    "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
}

SAVE_TRAINING_SAMPLES = True
HUMAN_REPLAY_DIR = "replay_human"
HUMAN_PGN_DIR = "human_games"


def action_to_move(action: int) -> chess.Move:
    frm = action // (64 * 5)
    to = (action // 5) % 64
    promo_idx = action % 5
    promo = None if promo_idx == 0 else {
        1: chess.KNIGHT,
        2: chess.BISHOP,
        3: chess.ROOK,
        4: chess.QUEEN,
    }[promo_idx]
    return chess.Move(frm, to, promotion=promo)


def save_pgn_from_moves(moves: list[chess.Move], result_str: str, out_dir: str, human_is_white: bool) -> str:
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


def save_human_shard(samples, out_dir: str):
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
        self,
        net,
        device="cpu",
        num_sims=50,
        save_training_samples: bool = SAVE_TRAINING_SAMPLES,
        human_replay_dir: str = HUMAN_REPLAY_DIR,
        human_pgn_dir: str = HUMAN_PGN_DIR,
        start_as_black: bool = False,
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

        self.square = 64
        self.margin = 10
        self.board_px = self.square * 8
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
        self.white_capture_slots = self._create_capture_grid(self.left_frame)
        tk.Label(self.right_frame, text="Black captured", font=("Arial", 12, "bold")).pack(anchor="n")
        self.black_capture_slots = self._create_capture_grid(self.right_frame)

        self.canvas = tk.Canvas(
            self.center_frame,
            width=self.board_px + 2 * self.margin,
            height=self.board_px + 2 * self.margin,
        )
        self.canvas.grid(row=0, column=0, columnspan=8, sticky="nsew")
        self.canvas.bind("<Configure>", self.on_canvas_resized)
        self.canvas.bind("<Button-1>", self.on_click)

        self.btn_new = tk.Button(self.center_frame, text="New Game", command=self.new_game)
        self.btn_new.grid(row=1, column=0, sticky="ew", padx=4, pady=(8, 0))

        self.btn_undo = tk.Button(self.center_frame, text="Undo (2 plies)", command=self.undo_two)
        self.btn_undo.grid(row=1, column=1, sticky="ew", padx=4, pady=(8, 0))

        self.btn_flip = tk.Button(self.center_frame, text="Flip Board", command=self.flip_board)
        self.btn_flip.grid(row=1, column=2, sticky="ew", padx=4, pady=(8, 0))

        self.play_as_black_var = tk.BooleanVar(value=bool(start_as_black))
        self.chk_black = tk.Checkbutton(
            self.center_frame,
            text="Play as Black",
            variable=self.play_as_black_var,
            command=self.on_toggle_side,
        )
        self.chk_black.grid(row=1, column=3, sticky="w", padx=4, pady=(8, 0))

        self.btn_prev = tk.Button(self.center_frame, text="◀ Prev", command=self.prev_move, state="disabled")
        self.btn_prev.grid(row=1, column=4, sticky="ew", padx=4, pady=(8, 0))

        self.btn_next = tk.Button(self.center_frame, text="Next ▶", command=self.next_move, state="disabled")
        self.btn_next.grid(row=1, column=5, sticky="ew", padx=4, pady=(8, 0))

        self.btn_latest = tk.Button(self.center_frame, text="Latest", command=self.go_latest, state="disabled")
        self.btn_latest.grid(row=1, column=6, sticky="ew", padx=4, pady=(8, 0))

        self.lbl = tk.Label(self.center_frame, text="", anchor="w")
        self.lbl.grid(row=2, column=0, columnspan=8, sticky="ew", pady=(8, 0))

        # --- Game state ---
        self.board = chess.Board()
        self.view_board = chess.Board()
        self.moves: list[chess.Move] = []
        self.selected_sq: int | None = None
        self.flipped = False
        self.sans: list[str] = []
        self.view_ply = 0

        # Record training: (state, pi, to_play_color)
        self.traj = []

        # Root for model MCTS (reused on model turns)
        self.root = Node(self.board.copy(stack=False))

        self.bind("<Left>", lambda e: self.prev_move())
        self.bind("<Right>", lambda e: self.next_move())

        self.update_label()
        self.update_nav_buttons()
        self.update_side_panels()
        self.draw()
        if not self.human_is_white:
            self.after(50, self.net_move)

    # -----------------------------
    # Side management
    # -----------------------------
    @property
    def human_is_white(self) -> bool:
        return not self.play_as_black_var.get()

    def on_toggle_side(self):
        # changing side mid-game can be confusing; we keep it simple:
        # toggling sets future new game side, and we update label.
        self.update_label()

    def update_label(self, extra: str = ""):
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

    # -----------------------------
    # Controls
    # -----------------------------
    def new_game(self):
        self.board = chess.Board()
        self.view_board = self.board.copy(stack=False)
        self.moves = []
        self.sans = []
        self.selected_sq = None
        self.view_ply = 0
        self.traj = []
        self.root = Node(self.board.copy(stack=False))
        self.update_label("New game.")
        self.update_nav_buttons()
        self.update_side_panels()
        self.draw()

        # If human is Black, model plays first (White)
        if not self.human_is_white:
            self.after(50, self.net_move)  # small delay so UI updates

    def flip_board(self):
        self.flipped = not self.flipped
        self.draw()

    def undo_two(self):
        if len(self.moves) >= 2:
            self.board.pop()
            self.board.pop()
            self.moves = self.moves[:-2]
            self.sans = self.sans[:-2]
            if len(self.traj) >= 2:
                self.traj = self.traj[:-2]
            self.root = Node(self.board.copy(stack=False))
            self.selected_sq = None
            self.set_view_ply(len(self.moves))
            self.update_label("Undid last two plies.")

    # -----------------------------
    # Input handling
    # -----------------------------
    def on_click(self, event):
        if self.board.is_game_over(claim_draw=True):
            return

        if self.view_ply != len(self.moves):
            self.update_label("You are viewing an earlier move. Press Latest to continue.")
            return

        # Human moves only on their color
        human_color = chess.WHITE if self.human_is_white else chess.BLACK
        if self.board.turn != human_color:
            return

        sq = self.pixel_to_square(event.x, event.y)
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

        # Create move; handle promotion via popup if needed
        move = chess.Move(from_sq, to_sq)
        if self.is_pawn_promotion(move):
            promo = self.ask_promotion_piece()
            if promo is None:
                # cancelled
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

        # Net responds
        self.after(10, self.net_move)

    def apply_human_move(self, move: chess.Move):
        # Record training target for human move: one-hot pi on the chosen move
        state = board_to_tensor(self.board)
        pi = np.zeros((ACTION_SIZE,), dtype=np.float32)
        pi[move_to_index(move)] = 1.0
        self.traj.append((state, pi, self.board.turn))

        self.sans.append(self.board.san(move))
        # Play move
        self.board.push(move)
        self.moves.append(move)

        # Reset root on human moves (simpler and correct)
        self.root = Node(self.board.copy(stack=False))
        self.set_view_ply(len(self.moves))

    # -----------------------------
    # Net move
    # -----------------------------
    def net_move(self):
        if self.board.is_game_over(claim_draw=True):
            return

        net_color = chess.BLACK if self.human_is_white else chess.WHITE
        if self.board.turn != net_color:
            return

        with torch.inference_mode():
            pi, action = mcts_policy_and_action(
                self.net,
                root=self.root,
                num_sims=self.num_sims,
                temperature=1e-6,
                device=self.device,
            )

        # Record training for net-to-move
        state = board_to_tensor(self.board)
        self.traj.append((state, pi.astype(np.float32), self.board.turn))

        mv = action_to_move(action)
        if mv not in self.board.legal_moves:
            mv = np.random.choice(list(self.board.legal_moves))

        self.sans.append(self.board.san(mv))
        self.board.push(mv)
        self.moves.append(mv)

        # Reuse tree for net
        self.root = reuse_root_after_action(self.root, action)
        self.root.board = self.board.copy(stack=False)

        if self.board.is_game_over(claim_draw=True):
            self.finish_game()
            return

        self.set_view_ply(len(self.moves))

    # -----------------------------
    # Finish + save
    # -----------------------------
    def finish_game(self):
        res = self.board.result(claim_draw=True)  # "1-0", "0-1", "1/2-1/2"

        if res == "1-0":
            z_white = 1.0
        elif res == "0-1":
            z_white = -1.0
        else:
            z_white = 0.0

        pgn_path = save_pgn_from_moves(
            self.moves,
            res,
            self.human_pgn_dir,
            human_is_white=self.human_is_white,
        )

        samples = []
        for state, pi, to_play in self.traj:
            v = z_white if to_play == chess.WHITE else -z_white
            samples.append((state, pi.astype(np.float32), float(v)))

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

    # -----------------------------
    # Helpers
    # -----------------------------
    def ask_promotion_piece(self):
        # Minimal promotion choice dialog
        win = tk.Toplevel(self)
        win.title("Promotion")
        win.resizable(False, False)
        choice = {"p": None}

        def pick(piece_type):
            choice["p"] = piece_type
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

    def is_pawn_promotion(self, mv: chess.Move) -> bool:
        piece = self.board.piece_at(mv.from_square)
        if piece is None or piece.piece_type != chess.PAWN:
            return False
        rank_to = chess.square_rank(mv.to_square)
        return (piece.color == chess.WHITE and rank_to == 7) or (piece.color == chess.BLACK and rank_to == 0)

    def pixel_to_square(self, x: int, y: int):
        x -= self.board_origin_x
        y -= self.board_origin_y
        if x < 0 or y < 0 or x >= self.board_px or y >= self.board_px:
            return None
        c = int(x // self.square)
        r = int(y // self.square)
        if self.flipped:
            c = 7 - c
            r = 7 - r
        file_ = c
        rank_ = 7 - r
        return chess.square(file_, rank_)

    def square_to_pixel_center(self, sq: int):
        file_ = chess.square_file(sq)
        rank_ = chess.square_rank(sq)
        r = 7 - rank_
        c = file_
        if self.flipped:
            r = 7 - r
            c = 7 - c
        x = self.board_origin_x + c * self.square + self.square / 2
        y = self.board_origin_y + r * self.square + self.square / 2
        return x, y

    def square_to_rect(self, sq: int):
        file_ = chess.square_file(sq)
        rank_ = chess.square_rank(sq)
        r = 7 - rank_
        c = file_
        if self.flipped:
            r = 7 - r
            c = 7 - c
        x0 = self.board_origin_x + c * self.square
        y0 = self.board_origin_y + r * self.square
        x1 = x0 + self.square
        y1 = y0 + self.square
        return x0, y0, x1, y1

    def _create_capture_grid(self, parent: tk.Widget) -> list[tk.Label]:
        box = tk.Frame(parent, relief="groove", bd=1, padx=4, pady=4)
        box.pack(anchor="n", fill="x", pady=(6, 0))
        for col in range(5):
            box.grid_columnconfigure(col, weight=1)
        slots: list[tk.Label] = []
        for row in range(3):
            box.grid_rowconfigure(row, weight=1)
            for col in range(5):
                lbl = tk.Label(box, text=" ", font=("Arial", 18), width=1, anchor="center")
                lbl.grid(row=row, column=col, sticky="nsew")
                slots.append(lbl)
        return slots

    def recompute_captures(self):
        b = chess.Board()
        white_taken = []
        black_taken = []
        for i in range(self.view_ply):
            mv = self.moves[i]
            mover_is_white = b.turn == chess.WHITE
            captured_symbol = None
            if b.is_en_passant(mv):
                captured_symbol = "p" if mover_is_white else "P"
            elif b.is_capture(mv):
                cap_piece = b.piece_at(mv.to_square)
                if cap_piece is not None:
                    captured_symbol = cap_piece.symbol()
            b.push(mv)
            if captured_symbol is not None:
                if mover_is_white:
                    white_taken.append(UNICODE_PIECE[captured_symbol])
                else:
                    black_taken.append(UNICODE_PIECE[captured_symbol])
        return white_taken, black_taken

    def update_side_panels(self):
        wcap, bcap = self.recompute_captures()
        for i, lbl in enumerate(self.white_capture_slots):
            lbl.config(text=wcap[i] if i < len(wcap) else " ")
        for i, lbl in enumerate(self.black_capture_slots):
            lbl.config(text=bcap[i] if i < len(bcap) else " ")

    def update_nav_buttons(self):
        self.btn_prev.config(state="normal" if self.view_ply > 0 else "disabled")
        self.btn_next.config(state="normal" if self.view_ply < len(self.moves) else "disabled")
        self.btn_latest.config(state="normal" if self.view_ply < len(self.moves) else "disabled")

    def set_view_ply(self, target_ply: int):
        target = max(0, min(target_ply, len(self.moves)))
        b = chess.Board()
        for i in range(target):
            b.push(self.moves[i])
        self.view_board = b
        self.view_ply = target
        if self.view_ply != len(self.moves):
            self.selected_sq = None
        self.update_nav_buttons()
        self.update_side_panels()
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
        self.canvas.delete("all")

        light = "#EEEED2"
        dark = "#769656"
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 50 or canvas_h < 50:
            canvas_w = self.board_px + 2 * self.margin
            canvas_h = self.board_px + 2 * self.margin

        reserve_left = max(16, int(canvas_w * 0.03))
        reserve_bottom = max(16, int(canvas_h * 0.03))
        board_px = max(
            160,
            min(
                canvas_w - 2 * self.margin - reserve_left,
                canvas_h - 2 * self.margin - reserve_bottom,
            ),
        )
        self.board_px = board_px
        self.square = board_px / 8
        self.coord_left_pad = max(12, int(self.square * 0.35))
        self.coord_bottom_pad = max(12, int(self.square * 0.35))
        self.board_origin_x = (canvas_w - board_px - self.coord_left_pad) / 2 + self.coord_left_pad
        self.board_origin_y = (canvas_h - board_px - self.coord_bottom_pad) / 2
        piece_font = max(18, int(self.square * 0.58))

        for r in range(8):
            for c in range(8):
                x0 = self.board_origin_x + c * self.square
                y0 = self.board_origin_y + r * self.square
                x1 = x0 + self.square
                y1 = y0 + self.square
                color = light if (r + c) % 2 == 0 else dark
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")

        if self.view_ply > 0:
            mv = self.moves[self.view_ply - 1]
            self._highlight_square(mv.from_square, outline="red")
            self._highlight_square(mv.to_square, outline="red")

        if self.selected_sq is not None and self.view_ply == len(self.moves):
            self._highlight_square(self.selected_sq, outline="blue")

        for sq, piece in self.view_board.piece_map().items():
            x, y = self.square_to_pixel_center(sq)
            ch = UNICODE_PIECE[piece.symbol()]
            self.canvas.create_text(x, y, text=ch, font=("Arial", piece_font), fill="#111")

        self._draw_coordinates()
        self.update_idletasks()

    def _draw_coordinates(self):
        label_font = ("Arial", max(9, int(self.square * 0.16)))
        text_color = "#333333"

        files = list("abcdefgh")
        if self.flipped:
            files = list(reversed(files))
        for c in range(8):
            x = self.board_origin_x + c * self.square + self.square / 2
            y = self.board_origin_y + self.board_px + self.coord_bottom_pad * 0.55
            self.canvas.create_text(x, y, text=files[c], font=label_font, fill=text_color)

        ranks = [str(r) for r in range(8, 0, -1)]
        if self.flipped:
            ranks = list(reversed(ranks))
        for r in range(8):
            x = self.board_origin_x - self.coord_left_pad * 0.55
            y = self.board_origin_y + r * self.square + self.square / 2
            self.canvas.create_text(x, y, text=ranks[r], font=label_font, fill=text_color)

    def _highlight_square(self, sq: int, outline="red"):
        x0, y0, x1, y1 = self.square_to_rect(sq)
        inset = max(2, int(self.square * 0.04))
        self.canvas.create_rectangle(
            x0 + inset,
            y0 + inset,
            x1 - inset,
            y1 - inset,
            outline=outline,
            width=max(2, int(self.square * 0.05)),
        )


def load_model(device="cpu", checkpoint_path: str = "checkpoint_latest.pt"):
    net = AlphaZeroNet(in_channels=18, channels=64, num_blocks=5).to(device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        payload = torch.load(checkpoint_path, map_location=device)
        if isinstance(payload, dict) and "model_state_dict" in payload:
            net.load_state_dict(payload["model_state_dict"])
            it = payload.get("iter")
            if it is not None:
                print(f"Loaded {checkpoint_path} (iter={it})")
            else:
                print(f"Loaded {checkpoint_path}")
        else:
            # Backward-compat: older checkpoints may contain weights directly.
            net.load_state_dict(payload)
            print(f"Loaded {checkpoint_path}")
    else:
        print(f"No checkpoint found at '{checkpoint_path}', using untrained model.")
    net.eval()
    return net


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play a local GUI game vs model.")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoint_latest.pt",
        help="Model checkpoint path to load.",
    )
    parser.add_argument("--num_sims", type=int, default=50, help="MCTS simulations per model move.")
    parser.add_argument("--play_as_black", action="store_true", help="Start game as Black.")
    parser.add_argument(
        "--save_training_samples",
        action="store_true",
        help="Save human-vs-model training shards to replay_human dir.",
    )
    parser.add_argument("--human_replay_dir", type=str, default=HUMAN_REPLAY_DIR)
    parser.add_argument("--human_pgn_dir", type=str, default=HUMAN_PGN_DIR)
    args = parser.parse_args()

    os.makedirs(args.human_pgn_dir, exist_ok=True)
    os.makedirs(args.human_replay_dir, exist_ok=True)

    net = load_model(device=args.device, checkpoint_path=args.checkpoint)

    app = PlayVsModel(
        net,
        device=args.device,
        num_sims=args.num_sims,
        save_training_samples=args.save_training_samples,
        human_replay_dir=args.human_replay_dir,
        human_pgn_dir=args.human_pgn_dir,
        start_as_black=args.play_as_black,
    )
    app.mainloop()
