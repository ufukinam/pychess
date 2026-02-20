from __future__ import annotations

import argparse
import os
import glob
import tkinter as tk
from tkinter import filedialog, messagebox

import chess
import chess.pgn


UNICODE_PIECE = {
    "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
    "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
}


class PGNViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Simple PGN Viewer")
        self.resizable(True, True)
        self.minsize(720, 520)

        # --- Board geometry ---
        self.square = 64
        self.margin = 10
        self.board_px = self.square * 8
        self.board_origin_x = self.margin
        self.board_origin_y = self.margin
        self._resize_after_id: str | None = None
        self.layout_mode = "wide"
        self.narrow_layout_width = 980

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1, minsize=170)
        self.grid_columnconfigure(1, weight=4, minsize=560)
        self.grid_columnconfigure(2, weight=1, minsize=170)

        # --- Layout frames (left captures | board | right captures) ---
        self.left_frame = tk.Frame(self)
        self.left_frame.grid(row=0, column=0, padx=(10, 6), pady=10, sticky="nsew")
        self.left_frame.bind("<Configure>", self.on_side_resized)

        self.center_frame = tk.Frame(self)
        self.center_frame.grid(row=0, column=1, padx=6, pady=10, sticky="nsew")
        self.center_frame.grid_rowconfigure(0, weight=1)
        for col in range(5):
            self.center_frame.grid_columnconfigure(col, weight=1)

        self.right_frame = tk.Frame(self)
        self.right_frame.grid(row=0, column=2, padx=(6, 10), pady=10, sticky="nsew")
        self.right_frame.bind("<Configure>", self.on_side_resized)

        # --- Left captured pieces (captured BY White) ---
        tk.Label(self.left_frame, text="White captured", font=("Arial", 12, "bold")).pack(anchor="n")
        self.white_capture_slots = self._create_capture_grid(self.left_frame)

        # --- Right captured pieces (captured BY Black) ---
        tk.Label(self.right_frame, text="Black captured", font=("Arial", 12, "bold")).pack(anchor="n")
        self.black_capture_slots = self._create_capture_grid(self.right_frame)

        # --- Board canvas ---
        self.canvas = tk.Canvas(
            self.center_frame,
            width=self.board_px + 2 * self.margin,
            height=self.board_px + 2 * self.margin
        )
        self.canvas.grid(row=0, column=0, columnspan=5, sticky="nsew")
        self.canvas.bind("<Configure>", self.on_canvas_resized)
        self.center_frame.bind("<Configure>", self.on_center_resized)

        # --- Controls ---
        self.btn_open = tk.Button(self.center_frame, text="Open PGN", command=self.open_pgn)
        self.btn_open.grid(row=1, column=0, sticky="ew", padx=4, pady=(8, 0))

        self.btn_latest = tk.Button(self.center_frame, text="Load Latest", command=self.load_latest)
        self.btn_latest.grid(row=1, column=1, sticky="ew", padx=4, pady=(8, 0))

        self.btn_prev = tk.Button(self.center_frame, text="◀ Prev", command=self.prev_move, state="disabled")
        self.btn_prev.grid(row=1, column=2, sticky="ew", padx=4, pady=(8, 0))

        self.btn_next = tk.Button(self.center_frame, text="Next ▶", command=self.next_move, state="disabled")
        self.btn_next.grid(row=1, column=3, sticky="ew", padx=4, pady=(8, 0))

        self.btn_reset = tk.Button(self.center_frame, text="Reset", command=self.reset_game, state="disabled")
        self.btn_reset.grid(row=1, column=4, sticky="ew", padx=4, pady=(8, 0))

        # --- Status + Total moves ---
        self.lbl_status = tk.Label(self.center_frame, text="Open a PGN to begin.", anchor="w")
        self.lbl_status.grid(row=2, column=0, columnspan=5, sticky="ew", pady=(8, 0))

        self.lbl_total = tk.Label(self.center_frame, text="", anchor="w")
        self.lbl_total.grid(row=3, column=0, columnspan=5, sticky="ew", pady=(2, 0))

        # --- Slider (ply navigation) ---
        self._updating_slider = False
        self.slider = tk.Scale(
            self.center_frame,
            from_=0,
            to=0,
            orient="horizontal",
            showvalue=True,
            command=self.on_slider_changed,
        )
        self.slider.grid(row=4, column=0, columnspan=5, sticky="ew", pady=(8, 0))

        # --- Game state ---
        self.game: chess.pgn.Game | None = None
        self.moves: list[chess.Move] = []
        self.sans: list[str] = []
        self.board = chess.Board()
        self.ply_index = 0  # plies applied (0 = start)

        # keyboard shortcuts
        self.bind("<Left>", lambda e: self.prev_move())
        self.bind("<Right>", lambda e: self.next_move())
        self.bind("<Configure>", self.on_window_resized)

        self.apply_layout_mode(force=True)
        self.draw()
        self.update_side_panels()
        self.update_total_label()
        self.update_slider_range()
        self.update_capture_slot_layout()

    # -----------------------------
    # LOAD FUNCTIONS
    # -----------------------------
    def open_pgn(self):
        path = filedialog.askopenfilename(
            title="Select PGN file",
            filetypes=[("PGN files", "*.pgn"), ("All files", "*.*")]
        )
        if path:
            self.load_pgn_file(path)

    def load_latest(self):
        games_dir = "games"
        if not os.path.exists(games_dir):
            messagebox.showinfo("Info", "No 'games' folder found.")
            return

        pgn_files = glob.glob(os.path.join(games_dir, "*.pgn"))
        if not pgn_files:
            messagebox.showinfo("Info", "No PGN files found in 'games' folder.")
            return

        latest_file = max(pgn_files, key=os.path.getmtime)
        self.load_pgn_file(latest_file)

    def load_pgn_file(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                game = chess.pgn.read_game(f)
            if game is None:
                raise ValueError("No game found in PGN.")

            board = game.board()
            moves = []
            sans = []
            for mv in game.mainline_moves():
                sans.append(board.san(mv))
                moves.append(mv)
                board.push(mv)

            self.game = game
            self.moves = moves
            self.sans = sans

            # jump to start
            self.set_ply_index(0)

            self.btn_reset.config(state="normal")
            self.btn_next.config(state="normal" if self.moves else "disabled")
            self.btn_prev.config(state="disabled")

            self.update_slider_range()
            self.update_status()
            self.update_total_label()
            self.update_side_panels()
            self.draw()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load PGN:\n{e}")

    # -----------------------------
    # Slider
    # -----------------------------
    def update_slider_range(self):
        max_ply = len(self.moves) if self.game is not None else 0
        self._updating_slider = True
        self.slider.config(from_=0, to=max_ply)
        self.slider.set(self.ply_index)
        self._updating_slider = False

    def on_center_resized(self, _event=None):
        wrap = max(200, self.center_frame.winfo_width() - 10)
        self.lbl_status.config(wraplength=wrap)
        self.lbl_total.config(wraplength=wrap)

    def on_canvas_resized(self, _event=None):
        if self._resize_after_id is not None:
            self.after_cancel(self._resize_after_id)
        self._resize_after_id = self.after(30, self.draw)

    def on_window_resized(self, event):
        if event.widget is not self:
            return
        self.apply_layout_mode()

    def apply_layout_mode(self, force: bool = False):
        width = self.winfo_width()
        if width <= 1:
            width = self.winfo_reqwidth()

        target = "narrow" if width < self.narrow_layout_width else "wide"
        if not force and target == self.layout_mode:
            return
        self.layout_mode = target

        if target == "wide":
            self.grid_rowconfigure(0, weight=1)
            self.grid_rowconfigure(1, weight=0)
            self.grid_columnconfigure(0, weight=1, minsize=170)
            self.grid_columnconfigure(1, weight=4, minsize=560)
            self.grid_columnconfigure(2, weight=1, minsize=170)

            self.left_frame.grid_configure(row=0, column=0, padx=(10, 6), pady=10, sticky="nsew")
            self.center_frame.grid_configure(row=0, column=1, columnspan=1, padx=6, pady=10, sticky="nsew")
            self.right_frame.grid_configure(row=0, column=2, padx=(6, 10), pady=10, sticky="nsew")
        else:
            self.grid_rowconfigure(0, weight=1)
            self.grid_rowconfigure(1, weight=0)
            self.grid_columnconfigure(0, weight=1, minsize=200)
            self.grid_columnconfigure(1, weight=1, minsize=200)
            self.grid_columnconfigure(2, weight=0, minsize=0)

            self.center_frame.grid_configure(row=0, column=0, columnspan=2, padx=8, pady=(10, 6), sticky="nsew")
            self.left_frame.grid_configure(row=1, column=0, padx=(10, 6), pady=(0, 10), sticky="nsew")
            self.right_frame.grid_configure(row=1, column=1, padx=(6, 10), pady=(0, 10), sticky="nsew")

        self.update_capture_slot_layout()

    def on_slider_changed(self, value_str: str):
        if self._updating_slider:
            return
        try:
            target = int(float(value_str))
        except ValueError:
            return
        self.set_ply_index(target)

    # -----------------------------
    # Navigation helpers
    # -----------------------------
    def set_ply_index(self, target_ply: int):
        if self.game is None:
            return

        target_ply = max(0, min(target_ply, len(self.moves)))

        # rebuild board from start to target ply (simple + correct)
        b = self.game.board()
        for i in range(target_ply):
            b.push(self.moves[i])

        self.board = b
        self.ply_index = target_ply

        self.update_buttons()
        self.update_status()
        self.update_total_label()
        self.update_side_panels()

        self._updating_slider = True
        self.slider.set(self.ply_index)
        self._updating_slider = False

        self.draw()

    def reset_game(self):
        if self.game is None:
            return
        self.set_ply_index(0)

    def next_move(self):
        if self.game is None or self.ply_index >= len(self.moves):
            return
        self.set_ply_index(self.ply_index + 1)

    def prev_move(self):
        if self.game is None or self.ply_index <= 0:
            return
        self.set_ply_index(self.ply_index - 1)

    def update_buttons(self):
        if self.game is None:
            self.btn_prev.config(state="disabled")
            self.btn_next.config(state="disabled")
            return
        self.btn_prev.config(state="normal" if self.ply_index > 0 else "disabled")
        self.btn_next.config(state="normal" if self.ply_index < len(self.moves) else "disabled")

    # -----------------------------
    # CAPTURED PIECES (recompute up to ply_index)
    # -----------------------------
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

    def on_side_resized(self, _event=None):
        self.update_capture_slot_layout()

    def update_capture_slot_layout(self):
        panel_w = min(self.left_frame.winfo_width(), self.right_frame.winfo_width())
        if panel_w <= 1:
            return

        # Keep 5 columns of piece glyphs visible in narrow side panels.
        cell_w = max(14, int((panel_w - 16) / 5))
        font_size = max(10, min(24, int(cell_w * 0.72)))
        pad_x = max(0, int(cell_w * 0.08))
        pad_y = max(0, int(font_size * 0.06))

        for lbl in self.white_capture_slots + self.black_capture_slots:
            lbl.config(font=("Arial", font_size), padx=pad_x, pady=pad_y)

    def recompute_captures(self):
        """
        Returns (white_taken, black_taken)
        white_taken: unicode pieces captured BY White (black pieces taken)
        black_taken: unicode pieces captured BY Black (white pieces taken)
        """
        if self.game is None:
            return [], []

        b = self.game.board()

        white_taken = []  # black pieces captured by White
        black_taken = []  # white pieces captured by Black

        for i in range(self.ply_index):
            mv = self.moves[i]
            mover_is_white = b.turn == chess.WHITE  # before push

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

    # -----------------------------
    # UI UPDATE
    # -----------------------------
    def update_status(self):
        if self.game is None:
            self.lbl_status.config(text="Open a PGN to begin.")
            return

        headers = self.game.headers
        white = headers.get("White", "White")
        black = headers.get("Black", "Black")
        result = headers.get("Result", "*")

        if self.ply_index == 0:
            move_info = "Start position"
        else:
            san = self.sans[self.ply_index - 1]
            move_no = (self.ply_index + 1) // 2
            side = "White" if self.ply_index % 2 == 1 else "Black"
            move_info = f"Move {move_no} ({side}): {san}"

        self.lbl_status.config(text=f"{white} vs {black} | {result} || {move_info}")

    def update_total_label(self):
        if self.game is None:
            self.lbl_total.config(text="")
            return
        total_plies = len(self.moves)
        total_full_moves = (total_plies + 1) // 2
        self.lbl_total.config(text=f"Total moves: {total_full_moves} (plies: {total_plies})")

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

        board_px = max(160, min(canvas_w - 2 * self.margin, canvas_h - 2 * self.margin))
        self.square = board_px / 8
        self.board_origin_x = (canvas_w - board_px) / 2
        self.board_origin_y = (canvas_h - board_px) / 2
        piece_font = max(14, int(self.square * 0.58))
        highlight_width = max(2, int(self.square * 0.05))

        # squares
        for r in range(8):
            for c in range(8):
                x0 = self.board_origin_x + c * self.square
                y0 = self.board_origin_y + r * self.square
                x1 = x0 + self.square
                y1 = y0 + self.square
                color = light if (r + c) % 2 == 0 else dark
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")

        # pieces
        for sq, piece in self.board.piece_map().items():
            r = 7 - chess.square_rank(sq)
            c = chess.square_file(sq)
            x = self.board_origin_x + c * self.square + self.square / 2
            y = self.board_origin_y + r * self.square + self.square / 2
            ch = UNICODE_PIECE[piece.symbol()]
            self.canvas.create_text(x, y, text=ch, font=("Arial", piece_font))

        # last move highlight
        if self.game is not None and self.ply_index > 0:
            mv = self.moves[self.ply_index - 1]
            self.highlight_square(mv.from_square, highlight_width)
            self.highlight_square(mv.to_square, highlight_width)

    def highlight_square(self, sq: int, width: int = 3):
        r = 7 - chess.square_rank(sq)
        c = chess.square_file(sq)
        x0 = self.board_origin_x + c * self.square
        y0 = self.board_origin_y + r * self.square
        x1 = x0 + self.square
        y1 = y0 + self.square
        inset = max(2, int(self.square * 0.04))
        self.canvas.create_rectangle(x0 + inset, y0 + inset, x1 - inset, y1 - inset, outline="red", width=width)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local PGN viewer GUI.")
    parser.add_argument(
        "--pgn_path",
        type=str,
        default="",
        help="Optional PGN path to load immediately.",
    )
    parser.add_argument(
        "--load_latest",
        action="store_true",
        help="Load latest PGN from --games_dir on startup.",
    )
    parser.add_argument(
        "--games_dir",
        type=str,
        default="games",
        help="Directory used with --load_latest.",
    )
    args = parser.parse_args()

    app = PGNViewer()
    if args.pgn_path:
        app.load_pgn_file(args.pgn_path)
    elif args.load_latest:
        if os.path.exists(args.games_dir):
            pgn_files = glob.glob(os.path.join(args.games_dir, "*.pgn"))
            if pgn_files:
                app.load_pgn_file(max(pgn_files, key=os.path.getmtime))
    app.mainloop()
