from __future__ import annotations

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
        self.resizable(False, False)

        # --- Layout frames (left captures | board | right captures) ---
        self.left_frame = tk.Frame(self)
        self.left_frame.grid(row=0, column=0, padx=(10, 6), pady=10, sticky="n")

        self.center_frame = tk.Frame(self)
        self.center_frame.grid(row=0, column=1, padx=6, pady=10)

        self.right_frame = tk.Frame(self)
        self.right_frame.grid(row=0, column=2, padx=(6, 10), pady=10, sticky="n")

        # --- Left captured pieces (captured BY White) ---
        tk.Label(self.left_frame, text="White captured", font=("Arial", 12, "bold")).pack(anchor="n")
        self.lbl_white_captured = tk.Label(self.left_frame, text="", font=("Arial", 22), justify="left", width=8)
        self.lbl_white_captured.pack(anchor="n", pady=(6, 0))

        # --- Right captured pieces (captured BY Black) ---
        tk.Label(self.right_frame, text="Black captured", font=("Arial", 12, "bold")).pack(anchor="n")
        self.lbl_black_captured = tk.Label(self.right_frame, text="", font=("Arial", 22), justify="left", width=8)
        self.lbl_black_captured.pack(anchor="n", pady=(6, 0))

        # --- Board canvas ---
        self.square = 64
        self.margin = 10
        self.board_px = self.square * 8
        self.canvas = tk.Canvas(
            self.center_frame,
            width=self.board_px + 2 * self.margin,
            height=self.board_px + 2 * self.margin
        )
        self.canvas.grid(row=0, column=0, columnspan=5)

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
        self.lbl_status = tk.Label(self.center_frame, text="Open a PGN to begin.", anchor="w", width=90)
        self.lbl_status.grid(row=2, column=0, columnspan=5, sticky="w", pady=(8, 0))

        self.lbl_total = tk.Label(self.center_frame, text="", anchor="w", width=90)
        self.lbl_total.grid(row=3, column=0, columnspan=5, sticky="w", pady=(2, 0))

        # --- Slider (ply navigation) ---
        self._updating_slider = False
        self.slider = tk.Scale(
            self.center_frame,
            from_=0,
            to=0,
            orient="horizontal",
            length=self.board_px,
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

        self.draw()
        self.update_side_panels()
        self.update_total_label()
        self.update_slider_range()

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
    def recompute_captures(self):
        """
        Returns (white_captured_str, black_captured_str)
        white_captured_str: pieces captured BY White (black pieces taken)
        black_captured_str: pieces captured BY Black (white pieces taken)
        """
        if self.game is None:
            return "", ""

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

        def wrap_pieces(pieces, per_line=6):
            if not pieces:
                return ""
            lines = []
            for j in range(0, len(pieces), per_line):
                lines.append(" ".join(pieces[j:j + per_line]))
            return "\n".join(lines)

        return wrap_pieces(white_taken), wrap_pieces(black_taken)

    def update_side_panels(self):
        wcap, bcap = self.recompute_captures()
        self.lbl_white_captured.config(text=wcap if wcap else "—")
        self.lbl_black_captured.config(text=bcap if bcap else "—")

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
        self.canvas.delete("all")

        light = "#EEEED2"
        dark = "#769656"

        # squares
        for r in range(8):
            for c in range(8):
                x0 = self.margin + c * self.square
                y0 = self.margin + r * self.square
                x1 = x0 + self.square
                y1 = y0 + self.square
                color = light if (r + c) % 2 == 0 else dark
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")

        # pieces
        for sq, piece in self.board.piece_map().items():
            r = 7 - chess.square_rank(sq)
            c = chess.square_file(sq)
            x = self.margin + c * self.square + self.square / 2
            y = self.margin + r * self.square + self.square / 2
            ch = UNICODE_PIECE[piece.symbol()]
            self.canvas.create_text(x, y, text=ch, font=("Arial", 36))

        # last move highlight
        if self.game is not None and self.ply_index > 0:
            mv = self.moves[self.ply_index - 1]
            self.highlight_square(mv.from_square)
            self.highlight_square(mv.to_square)

    def highlight_square(self, sq: int):
        r = 7 - chess.square_rank(sq)
        c = chess.square_file(sq)
        x0 = self.margin + c * self.square
        y0 = self.margin + r * self.square
        x1 = x0 + self.square
        y1 = y0 + self.square
        self.canvas.create_rectangle(x0 + 2, y0 + 2, x1 - 2, y1 - 2, outline="red", width=3)


if __name__ == "__main__":
    app = PGNViewer()
    app.mainloop()
