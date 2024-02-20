error_messages = {"move_unavailable": "Error: This is not a legal movement. Try again."}

initial_color = []

colors = ["w", "b"]
'ccc'
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
    "q": "Queen"
}
