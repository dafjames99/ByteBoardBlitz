from chess_types import Piece, Color, Config
from utils import *

class BitBoard:
    def __init__(self, squares: list[str] | list[int] | int = 0):
        if isinstance(squares, int):
            self.bb = squares
        elif isinstance(squares, list):
            if isinstance(squares[0], str):
                self.bb = BitBoard.from_fr(squares)
            elif isinstance(squares[0], int):
                self.bb = BitBoard.from_index(squares)
        else:
            self.bb = 0

    @staticmethod
    def from_index(squares: list[int]) -> int:
        bb = 0
        for idx in squares:
            bb |= 1 << idx
        return bb

    @staticmethod
    def from_fr(frs: list[str]) -> int:
        return BitBoard.from_index(fr_to_index(frs))
    
    def __repr__(self):
        return f"<BitBoard: {bin(self.bb)}>"

    def __int__(self):
        return self.bb
    
    def __contains__(self, square: int) -> bool:
        """Check if a square index is in the bitboard."""
        return bool(self.bb & (1 << square))

    def __and__(self, other):
        return BitBoard(self.bb & int(other))

    def __or__(self, other):
        return BitBoard(self.bb | int(other))

    def __xor__(self, other):
        return BitBoard(self.bb ^ int(other))

    def __invert__(self):
        return BitBoard(~self.bb & ((1 << 64) - 1))

    def __eq__(self, other):
        if isinstance(other, BitBoard):
            return self.bb == other.bb
        else:
            return self.bb == other

    def __str__(self):
        """
        Pretty-print a 64-bit bitboard as an 8x8 chessboard.
        Shows '1' for occupied bits, '.' for empty.
        Rank 8 is printed at the top, file 'a' on the left.
        """
        s = ''
        for rank in range(7, -1, -1):  # 7 → 0
            row = []
            for file in range(8):
                sq = rank * 8 + file  # index 0 = a1
                row.append("1" if (self.bb >> sq) & 1 else ".")
            s += f"\n{rank+1} {' '.join(row)}"
        s += "\n  A B C D E F G H"
        return s
    
    def add(self, square: str | int):
        """Set a square bit."""
        if isinstance(square, str):
            square = fr_to_index(square)
        self.bb |= 1 << (square)
    
    def remove(self, square: str | int):
        """Clear a square bit."""
        if isinstance(square, str):
            square = fr_to_index(square)
        self.bb &= ~(1 << (square))
        
    def to_frs(self) -> list[str]:
        """Return list of file-rank strings for all set bits."""
        return index_to_fr(self.to_index())
    
    def pretty_print(self) -> None:
        """
        Pretty-print a 64-bit bitboard as an 8x8 chessboard.
        Shows '1' for occupied bits, '.' for empty.
        Rank 8 is printed at the top, file 'a' on the left.
        """
        for rank in range(7, -1, -1):  # 7 → 0
            row = []
            for file in range(8):
                sq = rank * 8 + file  # index 0 = a1
                row.append("1" if (self.bb >> sq) & 1 else ".")
            print(f"{rank+1} {' '.join(row)}")
        print("  A B C D E F G H")

    @staticmethod
    def _and(*bbs) -> 'BitBoard':
        if isinstance(bbs[0], BitBoard):
            bbs = [int(b) for b in bbs]
        bb = bbs[0]
        for b in bbs:
            bb &= b
        return BitBoard(bb)

    @staticmethod
    def _or(*bbs) -> 'BitBoard':
        if isinstance(bbs[0], BitBoard):
            bbs = [int(b) for b in bbs]
        bb = bbs[0]
        for b in bbs:
            bb |= b
        return BitBoard(bb)

    def to_index(self) -> list[int]:
        """Return all set square indices in a bitboard."""
        return [i for i in range(64) if (self.bb >> i) & 1]

class PositionBoard(dict[Piece, BitBoard]):
    def __init__(self, color: Color, config: Config, piece_list: list[Piece]):
        super().__init__({})    
        for p in piece_list:
            if p in config.keys() and color in config[p].keys():
                self[p] = BitBoard(config[p][color])
            else:
                self[p] = BitBoard()
    def __getitem__(self, piece: str) -> BitBoard:
        return super().__getitem__(piece)

    def __setitem__(self, piece: str, bb: BitBoard):
        super().__setitem__(piece, bb)

    def all(self) -> BitBoard:
        """Union of all piece bitboards."""
        return BitBoard._or(*self.values())

