from chess_types import Config

MOVES: Config = { 
    "P": {'offsets': {
        'w': [8],
        'b': [-8]},
        'line': False,
    },
    "N": { 'offsets': {
        'w' : [10, 6, 15, 17, -10, -6, -15, -17],
        'b' : [10, 6, 15, 17, -10, -6, -15, -17]},
        'line': False,
    },
   "B":  { 'offsets': {
        'w': [-9, -7, 7, 9],
        'b': [-9, -7, 7, 9]},
        "line": True,
    },
    "R":  { 'offsets': {
        'w': [1, -1, 8, -8],
        'b': [1, -1, 8, -8]},
        "line": True,
    },
    "Q":  { 'offsets': {
        'w': [1, -1, 8, -8, -9, -7, 7, 9],
        'b': [1, -1, 8, -8, -9, -7, 7, 9]},
        "line": True,
    },
     "K":  { 'offsets': {
        "w": [1, -1, 8, -8, -9, -7, 7, 9],
        "b": [1, -1, 8, -8, -9, -7, 7, 9]},
        "line": False,
    },
}
POSITIONS: Config = { 
    "P": {
            "w": [f"{chr(97+i).upper()}2" for i in range(8)],
            "b": [f"{chr(97+i).upper()}7" for i in range(8)],
        },
    "N": {
            "w": ["B1", "G1"],
            "b": ["B8", "G8"],
        },
   "B": {
            "w": ["C1", "F1"],
            "b": ["C8", "F8"],
        },
    "R": {
            "w": ["A1", "H1"],
            "b": ["A8", "H8"],
        },
    "Q": 
        {"w": ["D1"], "b": ["D8"]},
    "K": {
        "w": ["E1"], "b": ["E8"]
    },
}


