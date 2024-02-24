error_messages = {"move_unavailable": "Error: This is not a legal movement. Try again."}

initial_color = []

letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
formalColors = {"w": "White", "b": "Black"}
colors = ["w", "b"]

startingPositions = {
    "P": [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7)],
    "N": [(0, 1), (0, 6)],
    "R": [(0, 0), (0, 7)],
    "B": [(0, 2), (0, 5)],
    "K": [(0, 4)],
    "Q": [(0, 3)],
    "p": [(6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7)],
    "n": [(7, 1), (7, 6)],
    "r": [(7, 0), (7, 7)],
    "b": [(7, 2), (7, 5)],
    "k": [(7, 4)],
    "q": [(7, 3)],
}

starting_bits = {
    "P": "0000000011111111000000000000000000000000000000000000000000000000",
    "N": "0100001000000000000000000000000000000000000000000000000000000000",
    "R": "1000000100000000000000000000000000000000000000000000000000000000",
    "B": "0010010000000000000000000000000000000000000000000000000000000000",
    "K": "0000100000000000000000000000000000000000000000000000000000000000",
    "Q": "0001000000000000000000000000000000000000000000000000000000000000",
    "p": "0000000000000000000000000000000000000000000000001111111100000000",
    "n": "0000000000000000000000000000000000000000000000000000000001000010",
    "r": "0000000000000000000000000000000000000000000000000000000010000001",
    "b": "0000000000000000000000000000000000000000000000000000000000100100",
    "k": "0000000000000000000000000000000000000000000000000000000000001000",
    "q": "0000000000000000000000000000000000000000000000000000000000010000",
}
startingSideIndices = {
    "w": {"q": {"R": 0, "N": 1, "B": 2}, "k": {"R": 7, "N": 6, "B": 5}},
    "b": {"q": {"r": 56, "n": 57, "b": 58}, "k": {"r": 63, "n": 62, "b": 61}},
}


movements = {
    "P": [[(1, 0)], [(1, -1), (1, 1)]],
    "N": [
        [(2, 1)],
        [(1, 2)],
        [(-2, 1)],
        [(-1, 2)],
        [(2, -1)],
        [(1, -2)],
        [(-2, -1)],
        [(-1, -2)],
    ],
    "R": [
        [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0)],
        [(-1, 0), (-2, 0), (-3, 0), (-4, 0), (-5, 0), (-6, 0), (-7, 0)],
        [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7)],
        [(0, -1), (0, -2), (0, -3), (0, -4), (0, -5), (0, -6), (0, -7)],
    ],
    "B": [
        [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],
        [(-1, 1), (-2, 2), (-3, 3), (-4, 4), (-5, 5), (-6, 6), (-7, 7)],
        [(1, -1), (2, -2), (3, -3), (4, -4), (5, -5), (6, -6), (7, -7)],
        [(-1, -1), (-2, -2), (-3, -3), (-4, -4), (-5, -5), (-6, -6), (-7, -7)],
    ],
    "Q": [
        [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0)],
        [(-1, 0), (-2, 0), (-3, 0), (-4, 0), (-5, 0), (-6, 0), (-7, 0)],
        [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7)],
        [(0, -1), (0, -2), (0, -3), (0, -4), (0, -5), (0, -6), (0, -7)],
        [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],
        [(-1, 1), (-2, 2), (-3, 3), (-4, 4), (-5, 5), (-6, 6), (-7, 7)],
        [(1, -1), (2, -2), (3, -3), (4, -4), (5, -5), (6, -6), (7, -7)],
        [(-1, -1), (-2, -2), (-3, -3), (-4, -4), (-5, -5), (-6, -6), (-7, -7)],
    ],
    "K": [
        [(0, 1)],
        [(0, -1)],
        [(1, 0)],
        [(-1, 0)],
        [(1, 1)],
        [(-1, -1)],
        [(-1, 1)],
        [(1, -1)],
    ],
    "p": [[(-1, 0)], [(-1, -1), (-1, 1)]],
    "n": [
        [(2, 1)],
        [(1, 2)],
        [(-2, 1)],
        [(-1, 2)],
        [(2, -1)],
        [(1, -2)],
        [(-2, -1)],
        [(-1, -2)],
    ],
    "r": [
        [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0)],
        [(-1, 0), (-2, 0), (-3, 0), (-4, 0), (-5, 0), (-6, 0), (-7, 0)],
        [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7)],
        [(0, -1), (0, -2), (0, -3), (0, -4), (0, -5), (0, -6), (0, -7)],
    ],
    "b": [
        [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],
        [(-1, 1), (-2, 2), (-3, 3), (-4, 4), (-5, 5), (-6, 6), (-7, 7)],
        [(1, -1), (2, -2), (3, -3), (4, -4), (5, -5), (6, -6), (7, -7)],
        [(-1, -1), (-2, -2), (-3, -3), (-4, -4), (-5, -5), (-6, -6), (-7, -7)],
    ],
    "q": [
        [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0)],
        [(-1, 0), (-2, 0), (-3, 0), (-4, 0), (-5, 0), (-6, 0), (-7, 0)],
        [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7)],
        [(0, -1), (0, -2), (0, -3), (0, -4), (0, -5), (0, -6), (0, -7)],
        [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],
        [(-1, 1), (-2, 2), (-3, 3), (-4, 4), (-5, 5), (-6, 6), (-7, 7)],
        [(1, -1), (2, -2), (3, -3), (4, -4), (5, -5), (6, -6), (7, -7)],
        [(-1, -1), (-2, -2), (-3, -3), (-4, -4), (-5, -5), (-6, -6), (-7, -7)],
    ],
    "k": [
        [(0, 1)],
        [(0, -1)],
        [(1, 0)],
        [(-1, 0)],
        [(1, 1)],
        [(-1, -1)],
        [(-1, 1)],
        [(1, -1)],
    ],
}

piece_types = ["P", "N", "R", "B", "K", "Q"]

dblBoard = {"P": [], "p": []}

dblMoves = [
    ("p", (48, 32)),
    ("p", (49, 33)),
    ("p", (50, 34)),
    ("p", (51, 35)),
    ("p", (52, 36)),
    ("p", (53, 37)),
    ("p", (54, 38)),
    ("p", (55, 39)),
    ("P", (8, 24)),
    ("P", (9, 25)),
    ("P", (10, 26)),
    ("P", (11, 27)),
    ("P", (12, 28)),
    ("P", (13, 29)),
    ("P", (14, 30)),
    ("P", (15, 31)),
]

formals = {
    "P": "Pawn",
    "N": "Knight",
    "R": "Rook",
    "B": "Bishop",
    "K": "King",
    "Q": "Queen",
    "p": "Pawn",
    "n": "Knight",
    "r": "Rook",
    "b": "Bishop",
    "k": "King",
    "q": "Queen",
}

indexMoves = {
'P': [[8], [7, 9]],
'N': [[17], [10], [-15], [-6], [15], [6], [-17], [-10]],
'R': [[8, 16, 24, 32, 40, 48, 56],
    [-8, -16, -24, -32, -40, -48, -56],
    [1, 2, 3, 4, 5, 6, 7],
    [-1, -2, -3, -4, -5, -6, -7]],
'B': [[9, 18, 27, 36, 45, 54, 63],
    [-7, -14, -21, -28, -35, -42, -49],
    [7, 14, 21, 28, 35, 42, 49],
    [-9, -18, -27, -36, -45, -54, -63]],
'Q': [[8, 16, 24, 32, 40, 48, 56],
    [-8, -16, -24, -32, -40, -48, -56],
    [1, 2, 3, 4, 5, 6, 7],
    [-1, -2, -3, -4, -5, -6, -7],
    [9, 18, 27, 36, 45, 54, 63],
    [-7, -14, -21, -28, -35, -42, -49],
    [7, 14, 21, 28, 35, 42, 49],
    [-9, -18, -27, -36, -45, -54, -63]],
'K': [[1], [-1], [8], [-8], [9], [-9], [-7], [7]],

'p': [[-8], [-9, -7]],
'n': [[17], [10], [-15], [-6], [15], [6], [-17], [-10]],
'r': [[8, 16, 24, 32, 40, 48, 56],
    [-8, -16, -24, -32, -40, -48, -56],
    [1, 2, 3, 4, 5, 6, 7],
    [-1, -2, -3, -4, -5, -6, -7]],
'b': [[9, 18, 27, 36, 45, 54, 63],
    [-7, -14, -21, -28, -35, -42, -49],
    [7, 14, 21, 28, 35, 42, 49],
    [-9, -18, -27, -36, -45, -54, -63]],
'q': [[8, 16, 24, 32, 40, 48, 56],
    [-8, -16, -24, -32, -40, -48, -56],
    [1, 2, 3, 4, 5, 6, 7],
    [-1, -2, -3, -4, -5, -6, -7],
    [9, 18, 27, 36, 45, 54, 63],
    [-7, -14, -21, -28, -35, -42, -49],
    [7, 14, 21, 28, 35, 42, 49],
    [-9, -18, -27, -36, -45, -54, -63]],
'k': [[1], [-1], [8], [-8], [9], [-9], [-7], [7]]
 }

white_pieces, black_pieces = ['K', 'Q', 'P', 'N', 'B', 'R'], ['k','q','p','n','b','r']


coloredPieces = {
    'w': {
        'Pawn': 'P',
        'Queen': 'Q',
        'King': 'K',
        'Bishop': 'B',
        'Knight': 'N',
        'Rook': 'R'
        },
    'b': {
        'Pawn': 'p',
        'Queen': 'q',
        'King': 'k',
        'Bishop': 'b',
        'Knight': 'n',
        'Rook': 'r'
        }
}