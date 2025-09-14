from chess_types import Config
from game import GameBoard
from config import POSITIONS
        
scholars_mate_moves = ['e4', 'a6', 'Bc4', 'a5', 'Qh5', 'a4', 'Qf7']
enpassant_test_w = ['a4', 'h6', 'a5', 'h5', 'c4', 'h4', 'c5', 'b5']
enpassant_test_b = ['a3', 'b5', 'a4', 'b4', 'a5', 'd5', 'a6', 'd4', 'c4']

castle_test = ['d3', 'a6', 'Be3', 'a5', 'Na3', 'a4', 'Qd2', 'b6']

promotion_test: Config = { 
    "P": {
            "w": ['a7'],
        },
     "K": {
        "w": ["E1"], "b": ["E8"]
    },
}

disambiguation_test: Config = { 
    "R": {
            "w": ['a1', 'h1'],
        },
     "K": {
        "w": ["e2"], "b": ["e5"]
    },
     "B": {'b': ['e1']}
}

state = {
    'ply': 1,
    'state': {
        'c': 'b',
        'double_pawn': None,
        'castle': {c: {s: False for s in ['Q', 'K']} for c in ['w', 'b']},
        'units': [],
        'san': '',
        'undo': False,
        
    }
}

if __name__ == "__main__":
    test = GameBoard()
    test.bulk_apply(['e4', 'e5', 'a3'])
    test.game_loop()
    # test.pretty_print()
    # [print(h) for h in test.history]
    # test.print_legal_moves()