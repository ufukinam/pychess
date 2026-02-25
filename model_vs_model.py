from __future__ import annotations

import argparse
import glob
import json
import os
import time
import tkinter as tk
from tkinter import messagebox

import chess
import chess.pgn
import torch

from mcts import Node, mcts_policy_and_action
from net import AlphaZeroNet
from encode import action_to_move, IN_CHANNELS
from chess_board_base import (
    BoardRenderer, create_capture_grid, recompute_captures, update_capture_display,
)

DEFAULT_PGN_DIR = "model_games"


def save_pgn_from_moves(moves, result_str, out_dir, white_name, black_name):
    os.makedirs(out_dir, exist_ok=True)
    game = chess.pgn.Game()
    game.headers["Event"] = "ModelVsModel"
    game.headers["Site"] = "Local"
    game.headers["Date"] = time.strftime("%Y.%m.%d")
    game.headers["White"] = white_name
    game.headers["Black"] = black_name
    game.headers["Result"] = result_str
    node = game
    for mv in moves:
        node = node.add_variation(mv)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"model_vs_model_{ts}.pgn")
    with open(path, "w", encoding="utf-8") as f:
        print(game, file=f, end="\n\n")
    return path


def discover_checkpoints():
    files: set[str] = set()
    for pattern in ("*.pt", "*.pth"):
        files.update(glob.glob(pattern))
    ordered: list[str] = []
    for preferred in ("checkpoint_latest.pt", "checkpoint_puzzle_latest.pt", "checkpoint_puzzle_best.pt"):
        if preferred in files:
            ordered.append(preferred)
            files.remove(preferred)
    ordered.extend(sorted(files))
    if not ordered:
        ordered = ["<untrained>"]
    return ordered


class ModelVsModelApp(tk.Tk):
    def __init__(self, device="cpu", num_sims=50, pgn_dir=DEFAULT_PGN_DIR,
                 move_delay_ms=400, channels=128, num_blocks=10):
        super().__init__()
        self.title("Model vs Model")
        self.resizable(True, True)
        self.minsize(980, 600)

        self.device = device
        self.num_sims = int(num_sims)
        self.pgn_dir = pgn_dir
        self.move_delay_ms = int(move_delay_ms)
        self.feedback_out_path = "feedback.jsonl"
        self.channels = channels
        self.num_blocks = num_blocks

        self.margin = 10
        self._resize_after_id: str | None = None
        self._autoplay_after_id: str | None = None

        self.model_cache: dict[str, AlphaZeroNet] = {}
        self.model_choices = discover_checkpoints()

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1, minsize=180)
        self.grid_columnconfigure(1, weight=4, minsize=620)
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

        self.renderer = BoardRenderer(self.canvas)

        self.white_model_var = tk.StringVar(value=self.model_choices[0])
        self.black_model_var = tk.StringVar(value=self.model_choices[0])

        tk.Label(self.center_frame, text="White model:").grid(row=1, column=0, sticky="e", padx=4, pady=(8, 0))
        self.opt_white = tk.OptionMenu(self.center_frame, self.white_model_var, *self.model_choices, command=lambda _: self.update_status())
        self.opt_white.grid(row=1, column=1, sticky="ew", padx=4, pady=(8, 0))
        tk.Label(self.center_frame, text="Black model:").grid(row=1, column=2, sticky="e", padx=4, pady=(8, 0))
        self.opt_black = tk.OptionMenu(self.center_frame, self.black_model_var, *self.model_choices, command=lambda _: self.update_status())
        self.opt_black.grid(row=1, column=3, sticky="ew", padx=4, pady=(8, 0))

        self.btn_play = tk.Button(self.center_frame, text="Play", command=self.start_autoplay)
        self.btn_play.grid(row=1, column=4, sticky="ew", padx=4, pady=(8, 0))
        self.btn_pause = tk.Button(self.center_frame, text="Pause", command=self.pause_autoplay)
        self.btn_pause.grid(row=1, column=5, sticky="ew", padx=4, pady=(8, 0))
        self.btn_step = tk.Button(self.center_frame, text="Step", command=self.step_once)
        self.btn_step.grid(row=1, column=6, sticky="ew", padx=4, pady=(8, 0))
        self.btn_new = tk.Button(self.center_frame, text="New Game", command=self.new_game)
        self.btn_new.grid(row=1, column=7, sticky="ew", padx=4, pady=(8, 0))
        self.btn_flip = tk.Button(self.center_frame, text="Flip Board", command=self.flip_board)
        self.btn_flip.grid(row=2, column=0, sticky="ew", padx=4, pady=(8, 0))
        self.btn_prev = tk.Button(self.center_frame, text="Prev", command=self.prev_move, state="disabled")
        self.btn_prev.grid(row=2, column=1, sticky="ew", padx=4, pady=(8, 0))
        self.btn_next = tk.Button(self.center_frame, text="Next", command=self.next_move, state="disabled")
        self.btn_next.grid(row=2, column=2, sticky="ew", padx=4, pady=(8, 0))
        self.btn_latest = tk.Button(self.center_frame, text="Latest", command=self.go_latest, state="disabled")
        self.btn_latest.grid(row=2, column=3, sticky="ew", padx=4, pady=(8, 0))
        self.btn_mark_bad = tk.Button(self.center_frame, text="Mark bad move", command=self.open_mark_bad_dialog)
        self.btn_mark_bad.grid(row=2, column=4, sticky="ew", padx=4, pady=(8, 0))

        self.lbl = tk.Label(self.center_frame, text="", anchor="w")
        self.lbl.grid(row=3, column=0, columnspan=8, sticky="ew", pady=(8, 0))

        self.board = chess.Board()
        self.view_board = self.board.copy(stack=False)
        self.moves: list[chess.Move] = []
        self.sans: list[str] = []
        self.view_ply = 0
        self.flipped = False
        self.autoplay = False
        self.board_history: list[chess.Board] = []

        self.roots = {
            chess.WHITE: Node(self.board.copy(stack=False)),
            chess.BLACK: Node(self.board.copy(stack=False)),
        }

        self.bind("<Left>", lambda _: self.prev_move())
        self.bind("<Right>", lambda _: self.next_move())

        self.update_nav_buttons()
        self._update_side_panels()
        self.update_status()
        self.draw()

    def selected_model_path(self, color):
        return self.white_model_var.get() if color == chess.WHITE else self.black_model_var.get()

    def load_model(self, checkpoint_path):
        if checkpoint_path in self.model_cache:
            return self.model_cache[checkpoint_path]
        net = AlphaZeroNet(in_channels=IN_CHANNELS, channels=self.channels,
                           num_blocks=self.num_blocks).to(self.device)
        if checkpoint_path != "<untrained>" and os.path.exists(checkpoint_path):
            payload = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if isinstance(payload, dict) and "model_state_dict" in payload:
                ch = payload.get("channels", self.channels)
                nb = payload.get("num_blocks", self.num_blocks)
                if ch != self.channels or nb != self.num_blocks:
                    net = AlphaZeroNet(in_channels=IN_CHANNELS, channels=ch, num_blocks=nb).to(self.device)
                net.load_state_dict(payload["model_state_dict"])
            else:
                net.load_state_dict(payload)
        else:
            print(f"No checkpoint at '{checkpoint_path}', using untrained model.")
        net.eval()
        self.model_cache[checkpoint_path] = net
        return net

    def _cancel_autoplay_job(self):
        if self._autoplay_after_id is not None:
            self.after_cancel(self._autoplay_after_id)
            self._autoplay_after_id = None

    def start_autoplay(self):
        if self.board.is_game_over(claim_draw=True):
            self.finish_game(show_popup=False)
            return
        self.autoplay = True
        self.go_latest()
        self._schedule_next_step(20)
        self.update_status("Autoplay running.")

    def pause_autoplay(self):
        self.autoplay = False
        self._cancel_autoplay_job()
        self.update_status("Autoplay paused.")

    def _schedule_next_step(self, delay_ms=None):
        self._cancel_autoplay_job()
        delay = self.move_delay_ms if delay_ms is None else int(delay_ms)
        self._autoplay_after_id = self.after(delay, self._play_step_autoplay)

    def _play_step_autoplay(self):
        self._autoplay_after_id = None
        if not self.autoplay:
            return
        self.play_one_move(schedule_next=True)

    def step_once(self):
        if self.autoplay:
            self.pause_autoplay()
        self.play_one_move(schedule_next=False)

    def play_one_move(self, schedule_next):
        if self.board.is_game_over(claim_draw=True):
            self.finish_game()
            return
        if self.view_ply != len(self.moves):
            self.go_latest()
        color = self.board.turn
        checkpoint = self.selected_model_path(color)
        net = self.load_model(checkpoint)

        root = self.roots[color]
        if root.board.fen() != self.board.fen():
            root = Node(self.board.copy(stack=False))
            self.roots[color] = root

        with torch.inference_mode():
            _pi, action = mcts_policy_and_action(
                net, root=root, num_sims=self.num_sims, temperature=1e-6,
                device=self.device, history=self.board_history,
            )
        move = action_to_move(action)
        if move not in self.board.legal_moves:
            move = next(iter(self.board.legal_moves))
        self.sans.append(self.board.san(move))
        self.board_history.append(self.board.copy(stack=False))
        self.board.push(move)
        self.moves.append(move)
        self.roots[chess.WHITE] = Node(self.board.copy(stack=False))
        self.roots[chess.BLACK] = Node(self.board.copy(stack=False))
        self.set_view_ply(len(self.moves))
        if self.board.is_game_over(claim_draw=True):
            self.finish_game()
            return
        if self.autoplay and schedule_next:
            self._schedule_next_step()

    def finish_game(self, show_popup=True):
        self.autoplay = False
        self._cancel_autoplay_job()
        res = self.board.result(claim_draw=True)
        white_name = os.path.basename(self.white_model_var.get())
        black_name = os.path.basename(self.black_model_var.get())
        pgn_path = save_pgn_from_moves(self.moves, res, self.pgn_dir, white_name, black_name)
        msg = f"Game over: {res}. Saved PGN: {pgn_path}"
        self.update_status(msg)
        if show_popup:
            messagebox.showinfo("Game finished", msg)

    def new_game(self):
        self.autoplay = False
        self._cancel_autoplay_job()
        self.board = chess.Board()
        self.view_board = self.board.copy(stack=False)
        self.moves = []
        self.sans = []
        self.view_ply = 0
        self.board_history = []
        self.roots = {
            chess.WHITE: Node(self.board.copy(stack=False)),
            chess.BLACK: Node(self.board.copy(stack=False)),
        }
        self.update_nav_buttons()
        self._update_side_panels()
        self.update_status("New game.")
        self.draw()

    def flip_board(self):
        self.flipped = not self.flipped
        self.renderer.flipped = self.flipped
        self.draw()

    def update_status(self, extra=""):
        if self.view_ply == 0:
            move_info = "Start position"
        else:
            san = self.sans[self.view_ply - 1] if self.view_ply - 1 < len(self.sans) else self.moves[self.view_ply - 1].uci()
            move_no = (self.view_ply + 1) // 2
            turn_side = "White" if self.view_ply % 2 == 1 else "Black"
            move_info = f"Move {move_no} ({turn_side}): {san}"
        turn = "White" if self.board.turn == chess.WHITE else "Black"
        base = (
            f"White={os.path.basename(self.white_model_var.get())} | "
            f"Black={os.path.basename(self.black_model_var.get())} | "
            f"To move={turn} | {move_info}"
        )
        if self.view_ply != len(self.moves):
            base += " (Inspect mode)"
        if extra:
            base += f". {extra}"
        self.lbl.config(text=base)

    def board_before_ply(self, ply):
        b = chess.Board()
        for i in range(max(0, ply)):
            b.push(self.moves[i])
        return b

    def open_mark_bad_dialog(self):
        if self.view_ply <= 0 or self.view_ply > len(self.moves):
            messagebox.showinfo("Mark bad move", "Navigate to a played move first (use Prev/Next).")
            return
        bad_idx = self.view_ply - 1
        board = self.board_before_ply(bad_idx)
        bad_move = self.moves[bad_idx]
        bad_san = board.san(bad_move)
        fen = board.fen()

        win = tk.Toplevel(self)
        win.title("Mark bad move")
        win.resizable(False, False)

        tk.Label(win, text=f"Ply {self.view_ply} bad move: {bad_san} ({bad_move.uci()})", anchor="w").grid(
            row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=(10, 4))
        tk.Label(win, text="Select better move:").grid(row=1, column=0, sticky="ne", padx=10, pady=4)

        list_frame = tk.Frame(win)
        list_frame.grid(row=1, column=1, sticky="w", padx=(0, 10), pady=4)
        good_move_list = tk.Listbox(list_frame, height=10, width=26, exportselection=False)
        scroll = tk.Scrollbar(list_frame, orient="vertical", command=good_move_list.yview)
        good_move_list.config(yscrollcommand=scroll.set)
        good_move_list.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")

        legal_moves = list(board.legal_moves)
        legal_display = sorted([(f"{board.san(mv):8s}  ({mv.uci()})", mv) for mv in legal_moves], key=lambda x: x[0])
        for txt, _ in legal_display:
            good_move_list.insert("end", txt)

        tk.Label(win, text="Confidence:").grid(row=2, column=0, sticky="e", padx=10, pady=4)
        conf_var = tk.StringVar(value="medium")
        tk.OptionMenu(win, conf_var, "low", "medium", "high").grid(row=2, column=1, sticky="w", padx=(0, 10), pady=4)

        tk.Label(win, text="Weight (optional):").grid(row=3, column=0, sticky="e", padx=10, pady=4)
        weight_var = tk.StringVar(value="")
        tk.Entry(win, textvariable=weight_var, width=22).grid(row=3, column=1, sticky="w", padx=(0, 10), pady=4)

        tk.Label(win, text="Output JSONL:").grid(row=4, column=0, sticky="e", padx=10, pady=4)
        out_var = tk.StringVar(value=self.feedback_out_path)
        tk.Entry(win, textvariable=out_var, width=42).grid(row=4, column=1, sticky="w", padx=(0, 10), pady=4)

        def submit():
            selected = good_move_list.curselection()
            if not selected:
                messagebox.showerror("Mark bad move", "Select a better move from the list.")
                return
            good_move = legal_display[selected[0]][1]
            if good_move == bad_move:
                messagebox.showerror("Mark bad move", "Preferred move must be different from bad move.")
                return
            row = {"fen": fen, "bad_move": bad_move.uci(), "good_move": good_move.uci(),
                   "confidence": conf_var.get(), "source": "model_vs_model_gui", "ply": int(self.view_ply)}
            wtxt = weight_var.get().strip()
            if wtxt:
                try:
                    row["weight"] = float(wtxt)
                except Exception:
                    messagebox.showerror("Mark bad move", f"Weight must be a number, got: {wtxt}")
                    return
            out_path = out_var.get().strip() or "feedback.jsonl"
            out_dir = os.path.dirname(out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            self.feedback_out_path = out_path
            self.update_status(f"Saved feedback ply {self.view_ply}: bad={bad_move.uci()} good={good_move.uci()} -> {out_path}")
            win.destroy()

        btn_row = tk.Frame(win)
        btn_row.grid(row=5, column=0, columnspan=2, sticky="e", padx=10, pady=(8, 10))
        tk.Button(btn_row, text="Save", width=10, command=submit).pack(side="left", padx=4)
        tk.Button(btn_row, text="Cancel", width=10, command=win.destroy).pack(side="left", padx=4)
        good_move_list.bind("<Double-Button-1>", lambda _: submit())
        if legal_display:
            good_move_list.selection_set(0)
        win.transient(self)
        win.grab_set()

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
        self.update_nav_buttons()
        self._update_side_panels()
        self.update_status()
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
        self.renderer.draw(self.view_board, last_move=last_move)
        self.update_idletasks()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a local GUI model-vs-model game in real time.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_sims", type=int, default=50)
    parser.add_argument("--move_delay_ms", type=int, default=400)
    parser.add_argument("--pgn_dir", type=str, default=DEFAULT_PGN_DIR)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--num_blocks", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.pgn_dir, exist_ok=True)
    app = ModelVsModelApp(
        device=args.device, num_sims=args.num_sims, pgn_dir=args.pgn_dir,
        move_delay_ms=args.move_delay_ms, channels=args.channels, num_blocks=args.num_blocks,
    )
    app.mainloop()
