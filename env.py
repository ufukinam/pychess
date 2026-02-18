from __future__ import annotations

import chess

class ChessEnv:
    def __init__(self) -> None:
        self.board = chess.Board()

    def reset(self) -> chess.Board:
        self.board.reset()
        return self.board

    def copy(self) -> "ChessEnv":
        e = ChessEnv()
        e.board = self.board.copy(stack=False)
        return e

    def legal_moves(self):
        return list(self.board.legal_moves)

    def push(self, move: chess.Move) -> None:
        self.board.push(move)

    def is_terminal(self) -> bool:
        return self.board.is_game_over(claim_draw=True)

    def result_value(self) -> float:
        """
        Returns game outcome from White's perspective:
          +1 white win, 0 draw, -1 black win
        """
        if not self.is_terminal():
            raise ValueError("Game not over yet")
        res = self.board.result(claim_draw=True)
        if res == "1-0":
            return 1.0
        if res == "0-1":
            return -1.0
        return 0.0
