from .utils import *
from .chess_types import Color, Piece, UnitDict, CastleVarDict, CastleSide,  MoveParam, MoveType

import copy
from contextlib import contextmanager

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

MOVES = { 
    "P": {
        'offsets': {
            'w': [8],
            'b': [-8]
        },
        'line': False,
    },
    "N":{ 
        'offsets': {
            'w' : [10, 6, 15, 17, -10, -6, -15, -17],
            'b' : [10, 6, 15, 17, -10, -6, -15, -17]
        },
        'line': False,
    },
    "B":{
        'offsets': {
            'w': [-9, -7, 7, 9],
            'b': [-9, -7, 7, 9]
        },
        "line": True,
    },
    "R":{
        'offsets': {
            'w': [1, -1, 8, -8],
            'b': [1, -1, 8, -8]
        },
        "line": True,
    },
    "Q":{
        'offsets': {
            'w': [1, -1, 8, -8, -9, -7, 7, 9],
            'b': [1, -1, 8, -8, -9, -7, 7, 9]
        },
        "line": True,
    },
    "K":{
        'offsets': {
            "w": [1, -1, 8, -8, -9, -7, 7, 9],
            "b": [1, -1, 8, -8, -9, -7, 7, 9]
        },
        "line": False,
    },
}

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
    def __init__(self, map: dict[Piece, BitBoard]):
        super().__init__(map)    
        
    def __getitem__(self, piece: str) -> BitBoard:
        return super().__getitem__(piece)

    def __setitem__(self, piece: str, bb: BitBoard):
        super().__setitem__(piece, bb)

    def all(self) -> BitBoard:
        """Union of all piece bitboards."""
        return BitBoard._or(*self.values())

class GameBoard:
    def __init__(self, initial_fen: str = START_FEN):
        
        self.move_config = MOVES
        self.pieces = list(self.move_config.keys())
        self.active_color = 'w'
        self.castle_rights = {c: {s: True for s in ['K', "Q"]} for c in ['w', 'b']}
        self.move_history = []
        self.pointer = -1
        self.en_passant = None
        
        pseudo_legal_mask = self.generate_pseudo_legal()
        
        self.moves = dict()
        self.moves['w'] = pseudo_legal_mask['w']
        self.moves['b'] = pseudo_legal_mask['b']

        self.pawn_attack = self.pawn_attacks()
        self.history = [initial_fen]
        
        self.legal_moves = None
        self.from_fen(initial_fen)

    def to_fen(self) -> str:
        """
        Generate a FEN string from the current board state.
        Only the first field (piece placement) is derived from self.positions;
        other fields are passed as args or placeholders.
        """
        posns = {i: '' for i in range(64)}
        for c in ['w', 'b']:
            for p in self.pieces:
                pos = self.positions[c][p].to_index()
                for x in pos:
                    posns[x] = p.lower() if c == 'b' else p.upper()

        fen_rows = []
        for rank_start in range(56, -1, -8):
            counter = 0
            row_str = ''
            for pos in range(rank_start, rank_start + 8):
                piece = posns[pos]
                if piece == '':
                    counter += 1
                else:
                    if counter > 0:
                        row_str += str(counter)
                        counter = 0
                    row_str += piece
            if counter > 0:
                row_str += str(counter)
            fen_rows.append(row_str)

        castling = ''
        for c in ['w', 'b']:
            for s in ['K', 'Q']:
                if self.castle_rights[c][s]:
                    if c == 'b':
                        castling += s.lower()
                    else:
                        castling += s
        
        if len(castling) == 0:
            castling = '-'
            
        piece_placement = "/".join(fen_rows)
        en_passant = index_to_fr(self.en_passant) if self.en_passant is not None else '-'
        fen_str = f"{piece_placement} {self.active_color} {castling} {en_passant} {self.halfmove} {self.fullmove}"
        return fen_str
    
    def from_fen(self, fen_str: str) -> 'GameBoard':
        """
        Load game state from a FEN string into self.gameboard.positions,
        active color, castling rights, en passant target, etc.
        """
        fields = fen_str.strip().split()
        if len(fields) != 6:
            raise ValueError("Invalid FEN: must have 6 fields")

        piece_placement, active_color, castling, en_passant, halfmove, fullmove = fields

        # Reset board
        self.positions = {c: PositionBoard({p: BitBoard() for p in self.pieces}) for c in ['w', 'b']}

        # Parse placement (ranks 8 → 1)
        ranks = piece_placement.split('/')
        if len(ranks) != 8:
            raise ValueError("Invalid FEN: must have 8 ranks")

        for rank_index, rank_str in enumerate(ranks):
            file_index = 0
            for ch in rank_str:
                if ch.isdigit():
                    file_index += int(ch)
                else:
                    c = 'w' if ch.isupper() else 'b'
                    p = ch.upper()
                    square = (7 - rank_index) * 8 + file_index  # convert rank/file to 0–63
                    self.positions[c][p].add(square)
                    file_index += 1

        # Active color
        self.active_color = active_color

        # Castling rights
        self.castle_rights = {
            'w': {'K': 'K' in castling, 'Q': 'Q' in castling},
            'b': {'K': 'k' in castling, 'Q': 'q' in castling}
        }

        # En passant target (convert algebraic square → index, or None) :: Square behind the double-moved pawn (e.g. if move is "e4", en_passant stores "e3")
        self.en_passant = None
        if en_passant != "-":
            file = ord(en_passant[0]) - ord('a')
            rank = int(en_passant[1]) - 1
            self.en_passant = rank * 8 + file

        # Halfmove + fullmove clocks
        self.halfmove = int(halfmove)
        self.fullmove = int(fullmove)
        self.legal_moves = self.generate_legal()
        # return self

    def generate_pseudo_legal(self):
        masks = {
            c: {
                p: {i: BitBoard() for i in range(64)}
                for p in self.pieces} for c in ['w', 'b']}
        for s in range(64):
            for p, config in self.move_config.items():
                if config['line']:
                    for c in ['w', 'b']:
                        masks[c][p][s] = {}
                    for offset in config['offsets']['w']:
                        for c in ['w', 'b']:
                            masks[c][p][s][offset] = []
                        dest = s
                        bb = BitBoard()
                        while True:
                            if (offset in [-9, -1, 7] and dest % 8 == 0) or \
                                (offset in [-7, 1, 9] and dest % 8 == 7) or \
                                (offset in [-9, -8, -7] and dest // 8 == 0) or \
                                (offset in [7, 8, 9] and dest // 8 == 7):
                                break
                            dest += offset
                            bb |= BitBoard([dest])
                            [masks[c][p][s][offset].append(copy.deepcopy(bb)) for c in ['w', 'b']]
                else:
                    for c, offsets in config['offsets'].items():
                        for offset in offsets:
                            if s + offset < 0 or s + offset > 63:
                                continue
                            if p == 'N':
                                if (s % 8 == 0 and offset in [-17, 15, 6, -10]) or \
                                    (s % 8 == 1 and  offset in [6, -10]) or \
                                    (s % 8 == 6 and  offset in [-6, 10]) or \
                                    (s % 8 == 7 and  offset in [17, -15, -6, 10]) or \
                                    (s // 8 == 0 and  offset in [-6, -10, -17, -15]) or \
                                    (s // 8 == 1 and  offset in [-17, -15]) or \
                                    (s // 8 == 6 and  offset in [6, 10]) or \
                                    (s // 8 == 7 and  offset in [6, 10, 17, 15]):
                                    continue
                            elif p == 'P':
                                if (s in [i for i in range(8, 16)] and c == 'w') or (s in [i for i in range(48, 56)] and c == 'b'):
                                    masks[c][p][s] |= BitBoard([s + 2*offset])
                            masks[c][p][s] |= BitBoard([s + offset])
        return masks
    
    def pawn_attacks(self) -> dict[str, dict[int, BitBoard]]:
        move_mask = {c :{i: BitBoard() for i in range(64)} for c in ['w', 'b']}
        for s in range(64):
            for c, ms in zip(['w', 'b'], [[7, 9], [-7, -9]]):
                e = 0 if c == 'b' else 7
                if s // 8 == e:
                    continue
                for m in ms:
                    if s % 8 == 0 and m in [-9, 7]: #Left edge
                        continue
                    if s % 8 == 7 and m in [9, -7]: #Right edge
                        continue
                    move_mask[c][s] |= BitBoard([s + m])
        return move_mask

    def get_occupied(self, c: Color) -> tuple[BitBoard, BitBoard, BitBoard]:
        '''All occupied for (own, opp, combined)'''
        all_own = self.positions[c].all()
        all_opp = self.positions[opp(c)].all()
        all_occ = all_own | all_opp
        return all_own, all_opp, all_occ
    
    def line_block(self, c: Color, p: Piece, s: int, own: BitBoard, opp: BitBoard, all: BitBoard) -> tuple[BitBoard, BitBoard]:
        bb = BitBoard()
        cb = BitBoard()
        for offset, bbs in self.moves[c][p][s].items():
            if not bbs:
                continue
            i = 0
            while True:
                if bbs[i] & all != BitBoard():
                    break
                if i == len(bbs) - 1:
                    break
                i += 1
            bb |= (bbs[i] & ~all)
            cb |= (bbs[i] & opp)
        return bb, cb
    
    def piece_basic_moves(self, c: Color, p: Piece, own_posns: BitBoard, opp_posns: BitBoard, all_posns: BitBoard, opp_k: int) -> tuple[list[tuple[tuple[Piece, int | None, int | None], str | None]], list[tuple[tuple[Piece, int | None, int | None], str | None]]]:
        
        indices = self.positions[c][p].to_index()
        moves, k_threats = [], []
        
        for i in indices:
            if self.move_config[p]['line']:
                mb, cb = self.line_block(c, p, i, own_posns, opp_posns, all_posns)
            else:
                mb = self.moves[c][p][i] & ~own_posns # Remove blocked-by-own
                
                if p == 'P':
                    mb = mb & ~opp_posns # Remove pawn-move to opp-square
                    
                    # -- Double pawn moves -- 
                    dbl_move = 16 if c == 'w' else -16
                    if i + dbl_move in mb:
                        moves.append(((p, i, i + dbl_move), 'double_pawn'))
                        mb.remove(i + dbl_move)
                    
                    # -- Pawn attack board -- 
                    cb = self.pawn_attack[c][i] & ~own_posns

                    # -- Enpassant move -- 
                    if self.en_passant is not None:
                        if self.en_passant in cb:
                            moves.append(((p, i, self.en_passant), 'enpassant'))
                            cb.remove(self.en_passant)
                    cb &= opp_posns
                
                # -- Knight and King moves -- 
                else:
                    cb = mb & opp_posns # Captures is move & opponent
                    mb &= ~opp_posns # Moves is move & non-opponent
                
                
            # -- Find k_threats -- 
            if opp_k in cb:
                k_threats.append(((p, i, opp_k), 'capture'))
                cb.remove(opp_k)
                
            for idx in mb.to_index():
                moves.append(((p, i, idx), None))
            for idx in cb.to_index():
                moves.append(((p, i, idx), 'capture'))
        return moves, k_threats
        
    def color_basic_moves(self, c: Color):
        opp_k = self.positions[opp(c)]['K'].to_index()[0]
        own_posns, opp_posns, all_posns = self.get_occupied(c)
        all_moves, all_k_threats = [], []
        for p in self.positions[c]:
            moves, k_threats = self.piece_basic_moves(c, p, own_posns, opp_posns, all_posns, opp_k)
            all_moves.extend(moves)
            all_k_threats.extend(k_threats)
        return all_moves, all_k_threats

    def is_under_check(self, c: Color, k_threats = None) -> bool:
        if k_threats is None:
            _, k_threats = self.color_basic_moves(opp(c))
        if len(k_threats) > 0:
            return True
        return False
    
    def legal_castle(self, c: Color) -> dict[CastleSide, bool]:
        legality = copy.deepcopy(self.castle_rights[c])
        if not (legality['Q'] or legality['K']):
            pass
        else:
            _, _, all_occ = self.get_occupied(c)
            k = 4 if c == 'w' else 60
            for side in ['Q', 'K']:
                
                r = (0 if c == 'w' else 56) if side == 'Q' else (7 if c == 'w' else 63)
                between = BitBoard([r+1, r+2, r+3]) if side == 'Q' else BitBoard([r-1, r-2])
                
                if legality[side]:
                    if (all_occ & between) == BitBoard():
                        for idx in between.to_index():
                            move_parameters=('K', k, idx)
                            with self.temp_move(move_parameters, None):
                                if self.is_under_check(c):
                                    legality[side] = False
                                    break
                    else:
                        legality[side] = False    
        return legality
        
    def generate_legal(self) -> list[tuple[tuple[Piece, int | None, int | None] | str, str | None]]:
        moves, _ = self.color_basic_moves(self.active_color)
        final = []
        # [print(mv) for mv in moves]
        for move_params, move_type in moves:
            with self.temp_move(move_params, move_type):
                if not self.is_under_check(self.active_color):
                    final.append((move_params, move_type))
        castle = self.legal_castle(self.active_color)
        [final.append((side, 'castle')) for side in castle.keys() if castle[side]]
        return final

    def unit(self, c: Color, p: Piece, old: int, new: int) -> UnitDict:
            return {
                'c': c,
                'p': p,
                'old': old,
                'new': new
            }
            
    def promote(self, c: Color, pos: int) -> tuple[UnitDict, UnitDict]:
        '''Select a new piece & instruct 2 units to insert into the state-dict'''
        while True:
            new_piece = input("Choose a piece to promote [Q, B, R, N]: ").upper()
            if new_piece in ['Q', 'B', 'R', 'N']:
                break
            print("Invalid choice.")
        return self.unit(c, 'P', pos, None), self.unit(c, new_piece, None, pos)
    
    def get_units(self, move_parameters: MoveParam, move_type: MoveType) -> tuple[list[UnitDict], CastleVarDict]: 
        
        units = []
        c = self.active_color
        castle_state = copy.deepcopy(self.castle_rights)
        
        if move_type == 'castle':
            k = 4 if c == 'w' else 60
            r = k - 4 if move_parameters == 'Q' else k + 3
            (k_offset, r_offset) = (-2, 3) if move_parameters == 'Q' else (2, -2)
            units.append(self.unit(c, 'K', k, k + k_offset))
            units.append(self.unit(c, 'R', r, r + r_offset))
            castle_state[c] = {'Q': False, 'K': False}

        else:    
            p, old, new = move_parameters
            units.append(self.unit(c, p, old, new))
            if p == 'K':
                castle_state[c] = {'Q': False, 'K': False}
            if p == "R":
                if (c == 'w' and old == 0) or (c == 'b' and old == 56):
                    castle_state[c]['Q'] = False
                elif (c == 'w' and old == 7) or (c == 'b' and old == 63):
                    castle_state[c]["K"] = False 
            if p == 'P':
                if ((c == 'w') and (new // 8 == 7)) or ((c == 'b') and (new // 8 == 0)):
                    old_unit, new_unit = self.promote(c, new)
                    units.append(old_unit)
                    units.append(new_unit)

            if move_type == 'capture':
                for q in self.pieces:
                    if new in self.positions[opp(c)][q]:
                        units.append(self.unit(opp(c), q, new, None))
                        break

            elif move_type == 'enpassant':
                cap = new - 8 if c == 'w' else new + 8
                units.append(self.unit(opp(c), p, cap, None))
        
        return units, castle_state
    
    def apply_units(self, units: list[UnitDict], reverse: bool = False) -> None:
        for unit in units:
            p, c = unit['p'], unit['c']
            if reverse:
                old, new = unit['new'], unit['old']
            else:
                old, new = unit['old'], unit['new']
            if old is not None:
                self.positions[c][p].remove(old)
            if new is not None:
                self.positions[c][p].add(new)
    
    @contextmanager
    def temp_move(self, move_parameters, move_type):
        units, _ = self.get_units(move_parameters, move_type)
        try:
            self.apply_units(units)
            yield
        finally:
            self.apply_units(units, reverse = True)
    
    def perm_move(self, move_paramters, move_type) -> None:
        
        if self.pointer != -1:
            self.history = self.history[:self.pointer + 1]
            self.move_history = self.move_history[:self.pointer + 1]
        
        if move_type == 'double_pawn':
            self.en_passant = move_paramters[2] - 8 if self.active_color == 'w' else move_paramters[2] + 8
        else:
            self.en_passant = None
        units, castle_state = self.get_units(move_paramters, move_type)
        self.apply_units(units)
        self.castle_rights = castle_state

        if self.active_color == 'b':
            self.fullmove += 1
        self.active_color = opp(self.active_color) 
        self.legal_moves = self.generate_legal()
        self.pointer = -1
    
    def apply_san(self, san: str) -> bool:
        move = self.parse_san(san)
        if move is None:
            print('Invalid san - no corresponding move.')
            return False
        self.perm_move(move[0], move[1])
        self.move_history.append(san)
        self.history.append(self.to_fen())
        return True
    
    def bulk_apply(self, san_list: list[str]):
        for san in san_list:
            self.apply_san(san)
    
    def undo(self) -> None:
        if len(self.history) == 1:
            print('No history to undo.')
            return
        self.pointer -= 1
        self.from_fen(self.history[self.pointer])

    def redo(self) -> None:
        if self.pointer == -1:
            print('No future to redo.')
            return
        self.pointer += 1
        self.from_fen(self.history[self.pointer])
    
    def get_san(self):
        return input('Enter move in SAN (Standard Algebraic Notation)) [ESC to return]: ')

    def coords_to_san(self, from_sq: int | str, to_sq: int | str):
        if isinstance(from_sq, str): from_sq = fr_to_index(from_sq)
        if isinstance(to_sq, str): to_sq = fr_to_index(to_sq)
    
        if (self.active_color == 'w' and from_sq == 4) or (self.active_color == 'b' and from_sq == 60):
            if to_sq == from_sq-2 and self.castle_rights[self.active_color]['Q']:
                return '0-0-0'
            elif to_sq == from_sq + 2 and self.castle_rights[self.active_color]['K']:
                return '0-0'
        for move_param, move_type in self.legal_moves:
            if move_type == 'castle':
                continue
            (p, old, new) = move_param
            if old == from_sq and new == to_sq:
                return self.move_to_san(move_param, move_type)

    def parse_san(self, san: str, c: Color = None):
        """Return the move tuple (move_parameters, move_type) for a SAN string."""
        # Strip check/mate symbols
        san = san.replace("+", "").replace("#", "")

        # --- Castling ---
        if san in ("O-O", "0-0"):
            return 'K', 'castle'
        if san in ("O-O-O", "0-0-0"):
            return 'Q', 'castle'

        # --- Promotion ---
        # promo = None
        # if "=" in san:
        #     san, promo = san.split("=")
        #     promo = promo.upper()

        # --- En passant marker ---
        enpassant = False
        if "ep" in san:
            enpassant = True
            san = san.replace("ep", "")
        san = san.strip()
        # --- Target square ---
        target_sq = san[-2:]
        target = fr_to_index(target_sq)
        # --- Piece type ---
        piece = "P"
        if san[0].isupper() and san[0] in "RNBQK":
            piece = san[0]
            san = san[1:]  # strip piece letter
        # --- Capture flag ---
        capture = "x" in san
        san = san.replace("x", "")

        # --- Disambiguation ---
        disambig = san[:-2]  # everything before square
        file_hint, rank_hint = None, None
        if disambig:
            if disambig[0].isalpha():
                file_hint = ord(disambig[0].lower()) - ord('a')
            if disambig.isdigit():
                rank_hint = int(disambig) - 1

        # --- Match against legal moves ---
        candidates = []
        for (sym, frm, to), move_type in self.legal_moves:
            if sym != piece:
                continue
            if to != target:
                continue
            if file_hint is not None and frm % 8 != file_hint:
                continue
            if rank_hint is not None and frm // 8 != rank_hint:
                continue
            if enpassant and move_type != "enpassant":
                # candidates.append(((sym, frm, to), move_type))
                continue
            if capture and move_type != "capture" and not enpassant:
                continue
            candidates.append(((sym, frm, to), move_type))

        if not candidates:
            return None

        if len(candidates) > 1:
            raise ValueError(f"Ambiguous SAN {san}, candidates={candidates}")

        return candidates[0]
    
    def move_to_san(
            self,
            move_params,
            move_type,
            check=False,
            mate=False
        ):
            """
            Convert a move tuple (move_params, move_type) back into SAN.
            Args:
                move_params: ('P'|'N'|'B'|'R'|'Q'|'K', from_idx, to_idx) OR 'K'/'Q' for castle
                move_type: 'castle' | 'capture' | 'enpassant' | 'double_pawn' | None
                c: color making the move
            """
            # --- Castling ---
            if move_type == "castle":
                return "O-O" if move_params == "K" else "O-O-O"

            piece, frm, to = move_params
            san = "" if piece == "P" else piece


            # --- Disambiguation ---
            # candidates = [
            #     (sym, f, t) for (sym, f, t), mt in legal_moves
            #     if sym == piece and t == to and f != frm
            # ]
            candidates = []
            for mp, mt in self.legal_moves:
                if mt == 'castle':
                    pass
                else:
                    if mp[0] == piece and mp[2] == to and mp[1] != frm:
                        candidates.append(mp)
            if candidates:
                if piece == 'P':
                    pass
                else:
                    same_file = any(f % 8 == frm % 8 for _, f, _ in candidates)
                    same_rank = any(f // 8 == frm // 8 for _, f, _ in candidates)
                    if not same_file:
                        san += chr(frm % 8 + ord('a'))
                    elif not same_rank:
                        san += str(frm // 8 + 1)
                    else:
                        san += chr(frm % 8 + ord('a')) + str(frm // 8 + 1)

            # --- Capture ---
            if move_type in ("capture", "enpassant"):
                if piece == "P":
                    # Pawn captures need file of origin
                    san += chr(frm % 8 + ord('a'))
                san += "x"

            # --- Target square ---
            san += index_to_fr(to).lower()

            # --- Promotion ---
            # if piece == "P":
            #     rank = to // 8
            #     if (self.active_color == "w" and rank == 7) or (self.active_color == "b" and rank == 0):
            #         for u in self.states.state()["units"]:
            #             if u["c"] == self.active_color and u["old"] is None and u["new"] == to:
            #                 san += "=P"
            #                 break

            if move_type == "enpassant":
                san += " ep"

            if check:
                san += "+"
            if mate:
                san += "#"

            return san
    
    def user_action(self):
        while True:
            action = input('\n--- ACTIONS ---\nM [View Moves]\tU [Undo]\tR [Redo] \tN [New Move]\tQ [Resign]\tH [History]\nChoice: ')
            if action in 'MURNQH':
                print('\n')
                break
            else:
                print('Invalid action.')
        return action

    def player_turn(self):
        while True:
            self.pretty_print()
            
            if self.is_gameover():
                return False

            print(f'\nMove {self.fullmove}. {formal_color(self.active_color)} to move.\n')

            while True:
                action = self.user_action()
                if action == "Q":
                    return 'resign'
                if action == 'M':
                    print([self.move_to_san(mv[0], mv[1]) for mv in self.legal_moves])
                if action == 'H':
                    self.print_move_history()
                if action == 'U':
                    print(" --- UNDO --- ")
                    self.undo()
                    return True
                if action == 'R':
                    print(" --- REDO --- ")
                    self.redo()
                    return True
                if action == 'N':
                    break
            while True:
                san = self.get_san()
                if san == 'ESC':
                    break
                move = self.apply_san(san)
                if move:
                    return True

    def game_loop(self):
        resign = False
        while True:
            t = self.player_turn()
            if t == 'resign':
                resign = True
                break
            elif not t:
                break
        self.print_verdict(resign = resign)
            
    def is_gameover(self):
        if len(self.legal_moves) == 0:
            if self.is_under_check(self.active_color):
                return True, 'checkmate'
            else:
                return True, 'stalemate'
        return False

    def print_legal_moves(self, raw: bool = False):
        if raw:
            [print(mv) for mv in self.legal_moves]
        else:
            print([self.move_to_san(mv[0], mv[1]) for mv in self.legal_moves])
    
    def print_verdict(self):
        if not self.is_gameover():
            print('Game ongoing')
        else:
            state = self.is_gameover()[1]
            s = f"Game ends by {state} on turn {self.fullmove}."
            if state == 'checkmate':
                s += f' {formal_color(opp(self.active_color))} wins.'
            print(s)

    def print_move_history(self):
        s = ''
        for i in range(0, len(self.move_history), 2):
            try:
                s += f'{i + 1}. {self.move_history[i]} {self.move_history[i + 1]}\n'
            except:
                s += f'{i + 1}. {self.move_history[i]}\n'
        print(s)
    
    def pretty_print(self):
        occ = {}
        occ['w'], occ['b'], _ = self.get_occupied('w')
        all = {i: '.' for i in range(64)}
        for c in ['w', 'b']:
            for sq in range(64):
                if sq in occ[c]:
                    for p in self.pieces:
                        if sq in self.positions[c][p]:
                            all[sq] = (p if c == 'w' else p.lower())
                            break
        for rank in range(7, -1, -1):  # 7 → 0
            row = []
            for file in range(8):
                sq = rank * 8 + file  # index 0 = a1
                row.append(all[sq])
            print(f"{rank+1} {' '.join(row)}")
        print("  A B C D E F G H")
