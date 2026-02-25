from __future__ import annotations

"""
Shared chess board rendering utilities for Tkinter GUIs.

Provides:
- UNICODE_PIECE constant
- BoardRenderer: stateful canvas-based board drawer
- Capture grid helpers
"""

import tkinter as tk
import chess


UNICODE_PIECE = {
    "P": "\u2659", "N": "\u2658", "B": "\u2657", "R": "\u2656", "Q": "\u2655", "K": "\u2654",
    "p": "\u265F", "n": "\u265E", "b": "\u265D", "r": "\u265C", "q": "\u265B", "k": "\u265A",
}


class BoardRenderer:
    """Renders a chess board on a Tk Canvas with coordinates and highlights."""

    def __init__(self, canvas: tk.Canvas):
        self.canvas = canvas
        self.flipped = False
        self.board_px = 512
        self.square = 64.0
        self.board_origin_x = 0.0
        self.board_origin_y = 0.0
        self.coord_left_pad = 12
        self.coord_bottom_pad = 12

    def compute_geometry(self, canvas_w: int, canvas_h: int, margin: int = 10):
        if canvas_w < 50 or canvas_h < 50:
            canvas_w = int(self.board_px) + 2 * margin
            canvas_h = int(self.board_px) + 2 * margin
        reserve_left = max(16, int(canvas_w * 0.03))
        reserve_bottom = max(16, int(canvas_h * 0.03))
        board_px = max(160, min(canvas_w - 2 * margin - reserve_left,
                                canvas_h - 2 * margin - reserve_bottom))
        self.board_px = board_px
        self.square = board_px / 8
        self.coord_left_pad = max(12, int(self.square * 0.35))
        self.coord_bottom_pad = max(12, int(self.square * 0.35))
        self.board_origin_x = (canvas_w - board_px - self.coord_left_pad) / 2 + self.coord_left_pad
        self.board_origin_y = (canvas_h - board_px - self.coord_bottom_pad) / 2

    def draw(self, board: chess.Board, last_move: chess.Move | None = None,
             selected_sq: int | None = None):
        self.canvas.delete("all")
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        self.compute_geometry(cw, ch)
        piece_font = max(18, int(self.square * 0.58))
        light, dark = "#EEEED2", "#769656"

        for r in range(8):
            for c in range(8):
                x0 = self.board_origin_x + c * self.square
                y0 = self.board_origin_y + r * self.square
                color = light if (r + c) % 2 == 0 else dark
                self.canvas.create_rectangle(x0, y0, x0 + self.square, y0 + self.square,
                                             fill=color, outline="")

        if last_move is not None:
            self._highlight_square(last_move.from_square, "red")
            self._highlight_square(last_move.to_square, "red")

        if selected_sq is not None:
            self._highlight_square(selected_sq, "blue")

        for sq, piece in board.piece_map().items():
            x, y = self.square_to_pixel_center(sq)
            self.canvas.create_text(x, y, text=UNICODE_PIECE[piece.symbol()],
                                    font=("Arial", piece_font), fill="#111")
        self._draw_coordinates()

    def _draw_coordinates(self):
        font = ("Arial", max(9, int(self.square * 0.16)))
        color = "#333333"
        files = list("abcdefgh")
        if self.flipped:
            files = list(reversed(files))
        for c in range(8):
            x = self.board_origin_x + c * self.square + self.square / 2
            y = self.board_origin_y + self.board_px + self.coord_bottom_pad * 0.55
            self.canvas.create_text(x, y, text=files[c], font=font, fill=color)
        ranks = [str(r) for r in range(8, 0, -1)]
        if self.flipped:
            ranks = list(reversed(ranks))
        for r in range(8):
            x = self.board_origin_x - self.coord_left_pad * 0.55
            y = self.board_origin_y + r * self.square + self.square / 2
            self.canvas.create_text(x, y, text=ranks[r], font=font, fill=color)

    def _highlight_square(self, sq: int, outline: str = "red"):
        x0, y0, x1, y1 = self.square_to_rect(sq)
        inset = max(2, int(self.square * 0.04))
        self.canvas.create_rectangle(x0 + inset, y0 + inset, x1 - inset, y1 - inset,
                                     outline=outline, width=max(2, int(self.square * 0.05)))

    def square_to_pixel_center(self, sq: int):
        f = chess.square_file(sq)
        r = 7 - chess.square_rank(sq)
        if self.flipped:
            r, f = 7 - r, 7 - f
        return (self.board_origin_x + f * self.square + self.square / 2,
                self.board_origin_y + r * self.square + self.square / 2)

    def square_to_rect(self, sq: int):
        f = chess.square_file(sq)
        r = 7 - chess.square_rank(sq)
        if self.flipped:
            r, f = 7 - r, 7 - f
        x0 = self.board_origin_x + f * self.square
        y0 = self.board_origin_y + r * self.square
        return x0, y0, x0 + self.square, y0 + self.square

    def pixel_to_square(self, x: int, y: int) -> int | None:
        x -= self.board_origin_x
        y -= self.board_origin_y
        if x < 0 or y < 0 or x >= self.board_px or y >= self.board_px:
            return None
        c = int(x // self.square)
        r = int(y // self.square)
        if self.flipped:
            c, r = 7 - c, 7 - r
        return chess.square(c, 7 - r)


# ---------- capture helpers ----------

def create_capture_grid(parent: tk.Widget, rows: int = 3, cols: int = 5) -> list[tk.Label]:
    box = tk.Frame(parent, relief="groove", bd=1, padx=4, pady=4)
    box.pack(anchor="n", fill="x", pady=(6, 0))
    for c in range(cols):
        box.grid_columnconfigure(c, weight=1)
    slots: list[tk.Label] = []
    for r in range(rows):
        box.grid_rowconfigure(r, weight=1)
        for c in range(cols):
            lbl = tk.Label(box, text=" ", font=("Arial", 18), width=1, anchor="center")
            lbl.grid(row=r, column=c, sticky="nsew")
            slots.append(lbl)
    return slots


def recompute_captures(moves: list[chess.Move], view_ply: int,
                       initial_board: chess.Board | None = None):
    """Return (white_taken, black_taken) unicode piece lists up to *view_ply*."""
    b = initial_board.copy() if initial_board else chess.Board()
    white_taken: list[str] = []
    black_taken: list[str] = []
    for i in range(view_ply):
        mv = moves[i]
        mover_white = b.turn == chess.WHITE
        cap_sym = None
        if b.is_en_passant(mv):
            cap_sym = "p" if mover_white else "P"
        elif b.is_capture(mv):
            cp = b.piece_at(mv.to_square)
            if cp is not None:
                cap_sym = cp.symbol()
        b.push(mv)
        if cap_sym is not None:
            (white_taken if mover_white else black_taken).append(UNICODE_PIECE[cap_sym])
    return white_taken, black_taken


def update_capture_display(slots: list[tk.Label], captures: list[str]):
    for i, lbl in enumerate(slots):
        lbl.config(text=captures[i] if i < len(captures) else " ")
