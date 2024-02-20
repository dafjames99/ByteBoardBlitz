import initial_vars as var
import game_mechanisms as gm

PAWN_DOUBLES, MOVEMENTS, STARTING_POSITIONS, PIECE_TYPES, COLORS, INIT_COLOR = (
    var.dblMoves,
    var.movements,
    var.startingPositions,
    var.piece_types,
    var.colors,
    var.initial_color,
)

Running = True
config = False, True, False
turnMoves, turnNumber, c, board, dblMoveIndex, whiteOnTop, audit = (
    [],
    0,
    "w",
    gm.bit_initialise(STARTING_POSITIONS),
    None,
    False,
    True,
)
while Running:
    # config = gm.gameConfig()
    gm.turnBlock(
        Running,
        turnMoves,
        turnNumber,
        c,
        board,
        audit,
        PAWN_DOUBLES,
        dblMoveIndex,
        whiteOnTop,
        config[0],
        config[1],
        config[2],
    )
print("Game Ended.")
