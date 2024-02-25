import mechanics as mc
import initial_vars as var

PIECE_VALUES = {
    "P": 1,
    "N": 3,
    "R": 5,
    "B": 3,
    "Q": 9,
    "p": -1,
    "n": -3,
    "r": -5,
    "b": -3,
    "q": -9
}

def kingLocation(board, piece):
    for i in range(64):
        if board[piece][i] == '1':
            return i

def boardValue(board):
    king = ['k', 'K']
    value = 0
    for piece in board:
        if piece not in king:
            for i in mc.bit_to_indices(piece, board):
                value += PIECE_VALUES[piece]
    return value



            