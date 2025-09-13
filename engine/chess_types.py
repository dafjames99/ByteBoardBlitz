from typing import Literal, TypeAlias, TypedDict, NotRequired

Color: TypeAlias = Literal['w', 'b']
Piece: TypeAlias = Literal['P', 'B', 'N', 'R', 'Q', 'K']

CastleSide: TypeAlias = Literal["W", "Q"]
EndStateTypes: TypeAlias = Literal['checkmate', 'stalemate', 'draw_by_repetition', 'draw']
StateDictKeys: TypeAlias = Literal["c", "castle", "double_pawn", "units", "undo"]

class PieceConfig(TypedDict):
    line: NotRequired[bool]
    w: list
    b: list
    
class Config(TypedDict):
    P: PieceConfig
    N: PieceConfig
    B: PieceConfig
    R: PieceConfig
    Q: PieceConfig
    K: PieceConfig

class CastleDict(TypedDict):
    Q: bool
    K: bool

class CastleVarDict(TypedDict):
    w: CastleDict
    b: CastleDict

class UnitDict(TypedDict):
    p: Piece
    c: Color
    old: int | None
    new: int | None

class StateDict(TypedDict):
    c: Color
    castle: CastleVarDict
    double_pawn: int | None
    units: list[UnitDict]
    undo: bool
    san: str

