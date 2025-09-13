import json
import copy

from bitboard import BitBoard, PositionBoard
from utils import *
from chess_types import Color, Piece, StateDict, UnitDict, CastleVarDict, EndStateTypes, StateDictKeys, CastleSide
from config import *

import copy
from typing import Literal
from contextlib import contextmanager

class StateManager(dict[int, StateDict]):
    def __init__(self, gameboard: 'GameBoard', initial_package: dict[str, int | StateDict] = None):
        """Define the initial state of the gameboard.
        Args:
            gameboard (GameBoard): associated gameboard.
            initial_package (dict[str, int  |  StateDict], optional): Takes a {ply: {c, units, double_pawn, castle, undo, san}} argument. Defaults to None (regular start).
        """        
        if initial_package is None:
            ply = 1
            starting_c = 'w'
            initial_state: StateDict = {
                'c': opp(starting_c),
                'units': [],
                'double_pawn': None,
                'castle': {c: {"Q": True, 'K': True} for c in ['w', 'b']},
                'undo': False, 
                'san': ''
            }
        else:
            ply = initial_package['ply']
            initial_state = initial_package['state']
        super().__init__(
            {ply: initial_state}
        )
        self.pointer = ply
        self.initial_pointer = ply
        self.end_state = None
        self.gameboard = gameboard
        
    def state(self, pointer: int | None = None) -> StateDict:
        if pointer is None:
            pointer = self.pointer    
        return self.get(pointer, None)
            
    def query(self, value: StateDictKeys, pointer: int | None = None) -> Color | CastleVarDict | int | None | list[UnitDict] | bool:
        state = self.state(pointer)
        if state is None:
            return None
        return state.get(value, None)
    
    def unit(self, c: Color, p: Piece, old: int, new: int) -> UnitDict:
            return {
                'c': c,
                'p': p,
                'old': old,
                'new': new
            }

    def new_state(self,
                  move_parameters: Literal['K', 'Q'] | tuple[Piece, int | None, int | None] = None,
                  move_type: Literal['castle', 'capture', 'enpassant', 'double_pawn'] = None,
                  san = None,
                  c: Color = None,
                  apply = False) -> StateDict:
        
        """Returns the state dict defined by the move_type & move_parameters. Output is ready for applying moves & appending to states-list.

        Args:
            c (Color): _description_: Color affecting the state - active color. 
            move_parameters ('K' | 'Q' | tuple[Piece, int  |  None, int  |  None]): _description_: Parameters for the move. If move_type is castle, params are either Q or K (side of castling). Else, move_parameters are (piece, old-position-index, new-position-index)
            move_type (['castle' |'capture' | 'enpassant' | 'double_pawn'], optional): _description_: Defaults to None (for a regular move).

        Returns:
            state: {
            'c': Color,
            'ply': pointer + 1
            'units': list[{'p': piece, 'old': int | None, 'new': int | None}],
            'vars': {
                'double_pawn': int | None,
                'castle': dict
            },
            'undo': False}
        """        
        # if san is not None:
        #     move_parameters, move_type = self.gameboard.parse_san(san)
        # else:
        #     san = self.gameboard.move_to_san(move_parameters, move_type, c)
        units = []
        castle_state = copy.deepcopy(self.query('castle'))
        if c is None:
            c = opp(self.query('c'))
        double_pawn = None
        
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
                if apply:
                    if ((c == 'w') and (new // 8 == 7)) or ((c == 'b') and (new // 8 == 0)):
                        old_unit, new_unit = self.promote(c, new)
                        units.append(old_unit)
                        units.append(new_unit)

            if move_type == 'capture':
                for q in self.gameboard.pieces:
                    if new in self.gameboard.positions[opp(c)][q]:
                        units.append(self.unit(opp(c), q, new, None))
                        break

            elif move_type == 'enpassant':
                cap = new - 8 if c == 'w' else new + 8
                units.append(self.unit(opp(c), p, cap, None))

            elif move_type == 'double_pawn':
                double_pawn = new

        state = {
            'c': c,
            'units': units,
            'double_pawn': double_pawn,
            'castle': castle_state,
            'undo': False,
            'san': san
        }
        return state
   
    def promote(self, c: Color, pos: int) -> tuple[UnitDict, UnitDict]:
        '''Select a new piece & instruct 2 units to insert into the state-dict'''
        while True:
            new_piece = input("Choose a piece to promote [Q, B, R, N]: ").upper()
            if new_piece in ['Q', 'B', 'R', 'N']:
                break
            print("Invalid choice.")
        return self.unit(c, 'P', pos, None), self.unit(c, new_piece, None, pos)

    def next_undo(self) -> int | None:
        '''Return index of first state in undo-chain.'''
        if len(self) == 1:
            return None
        for pointer in range(len(self)):
            if self.query('undo', pointer):
                return pointer - 1
        return len(self) - 1

    def undo(self, narrate = False) -> None:
        '''Add a state to the undo-chain. Units are reversed.'''
        if self.pointer == list(self.keys())[0]:
            print("No history to undo.")
            return
        state = self.state()
        self.gameboard.apply_units(state['units'], reverse = True, narrate = narrate)
        state['undo'] = True
        self.pointer -= 1
    
    def redo(self, narrate = False) -> None:
        '''Remove a state from the undo-chain. Units are re-applied.'''
        if self.pointer == list(self.keys())[-1]:
            print("No future to redo.")
            return
        state = self.state(self.pointer + 1)
        state['undo'] = False
        self.gameboard.apply_units(state['units'], narrate = narrate)
        self.pointer += 1
        
    def reset_branch(self) -> None:
        while True:
            try:
                self.pop(self.pointer + 1)
            except:
                break
        
    def apply_state(self, state: dict, narrate = False) -> None:
        '''Append to states-list & enact units.'''
        self.reset_branch()
        self.pointer += 1
        self[self.pointer] = state
        self.gameboard.apply_units(state['units'], narrate = narrate)

    @contextmanager
    def temp(self, state):
        try:
            self.gameboard.apply_units(state['units'])
            yield
        finally:
            self.gameboard.apply_units(state['units'], reverse = True)

    def end(self, result: EndStateTypes, winner: Color = None):
        
        if result == 'checkmate':
            print(f'Game over. {formal_color(winner)} wins by checkmate.')
        elif result == 'stalemate':
            print(f'Game Over. Draw by Stalemate.')
        elif result == 'draw-by_repetition':
            print(f'Game Over. Draw by Repetition.')
        elif result == 'draw':
            print(f'Game Over. Draw agreed.')
        
        self.end_state = {
            'ply': self.pointer - 1,
            'result': result,
            'winner': winner
        }


class GameBoard:
    def __init__(self, position_config: Config = None, initial_package: dict[str, int | StateDict] = None):
        self.move_config: Config = MOVES
        
        self.pos_config = position_config
        self.pieces = list(self.move_config.keys())
        
        self.positions = dict()
        if position_config is None:
            position_config = POSITIONS
        self.positions['w'] = PositionBoard(color = 'w', config=position_config, piece_list=self.pieces)
        self.positions['b'] = PositionBoard(color = 'b', config=position_config, piece_list=self.pieces)
        
        pseudo_legal_mask = self.generate_pseudo_legal()
        
        self.moves = dict()
        self.moves['w'] = pseudo_legal_mask['w']
        self.moves['b'] = pseudo_legal_mask['b']

        self.pawn_attack = self.pawn_attacks()
        
        self.states = StateManager(self, initial_package)
    
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
    
    # def piece_basic_moves(self, c: Color, p: Piece, own_posns: BitBoard, opp_posns: BitBoard, all_posns: BitBoard, dbl_pawn):
        
    #     opp_k = self.positions[opp(c)]['K'].to_index()[0]
    #     indices = self.positions[c][p].to_index()
        
    #     kb = BitBoard()
        
    #     for i in indices:
    #         if self.move_config[p]['line']:
    #             mb, cb = self.line_block(c, p, i, own_posns, opp_posns, all_posns)
    #         else:
    #             mb = BitBoard() | (self.moves[c][p][i] & ~own_posns) # Knight, King, pawn MOVE
                
    #             if p == 'P':
    #                 mb = mb & ~opp_posns
    #                 dbl_move = 16 if c == 'w' else -16
    #                 if i + dbl_move in mb:
    #                     moves.append(((p, i, i + dbl_move), 'double_pawn'))
    #                     mb.remove(i + dbl_move)
    #                 cb = self.pawn_attack[c][i]
    #                 cb &= ~own_posns
    #                 if dbl_pawn is not None:
    #                     offset = 8 if c == 'w' else -8
    #                     if (dbl_pawn + offset) in cb:
    #                         moves.append(((p, i, dbl_pawn + offset), 'enpassant'))
    #                         cb.remove(dbl_pawn + offset)
    #                 cb &= opp_posns
    #             else:
    #                 cb = mb & opp_posns
    #                 mb &= ~opp_posns
    #                 # mb &= ~own_posns 
    #                 # print(p, cb, mb)
                
    #         if opp_k in cb:
    #             kb = BitBoard([i])
    #             # k_threats.append(((p, i, opp_k_sq), 'capture'))
    #             cb.remove(opp_k_sq)
            
    def prune_blocked(self, c: Color) -> list[tuple[tuple[Piece, int | None, int | None] | str, str | None]]:
        '''Get non-blocked moves for color c. Output is list of (move_params, move_type) tuples (pseudo-legals).'''
        
        opp_k_sq = self.positions[opp(c)]['K'].to_index()[0]
        
        own_posns, opp_posns, all_posns = self.get_occupied(c)
        
        dbl_pawn = self.states.query('double_pawn')
        
        piece_indices = {
            p: self.positions[c][p].to_index()
            for p in self.pieces
        }
        
        moves = []
        k_threats = []
        
        for p, indices in piece_indices.items():
            for i in indices:
                if self.move_config[p]['line']:
                    mb, cb = self.line_block(c, p, i, own_posns, opp_posns, all_posns)
                else:
                    mb = BitBoard() | (self.moves[c][p][i] & ~own_posns) # Knight, King, pawn MOVE
                    
                    if p == 'P':
                        mb = mb & ~opp_posns
                        dbl_move = 16 if c == 'w' else -16
                        if i + dbl_move in mb:
                            moves.append(((p, i, i + dbl_move), 'double_pawn'))
                            mb.remove(i + dbl_move)
                        cb = self.pawn_attack[c][i]
                        cb &= ~own_posns
                        if dbl_pawn is not None:
                            offset = 8 if c == 'w' else -8
                            if (dbl_pawn + offset) in cb:
                                moves.append(((p, i, dbl_pawn + offset), 'enpassant'))
                                cb.remove(dbl_pawn + offset)
                        cb &= opp_posns
                    else:
                        cb = mb & opp_posns
                        mb &= ~opp_posns
                        # mb &= ~own_posns 
                        # print(p, cb, mb)
                
                if opp_k_sq in cb:
                    k_threats.append(((p, i, opp_k_sq), 'capture'))
                    cb.remove(opp_k_sq)
                [moves.append(((p, i, idx), None)) for idx in mb.to_index()]
                [moves.append(((p, i, idx), 'capture')) for idx in cb.to_index()]
        return moves, k_threats

    def apply_units(self, units: list[UnitDict], reverse: bool = False, narrate = False) -> None:
        for unit in units:
            p, c = unit['p'], unit['c']
            if reverse:
                old, new = unit['new'], unit['old']
            else:
                old, new = unit['old'], unit['new']
            if old is not None:
                if narrate: print(f'{c} {p} remove at {index_to_fr(old)}')
                self.positions[c][p].remove(old)
            if new is not None:
                if narrate: print(f'{c} {p} add at {index_to_fr(new)}')
                self.positions[c][p].add(new)

    def is_under_check(self, c: Color, k_threats = None) -> bool:
        if k_threats is None:
            _, k_threats = self.prune_blocked(opp(c))
        if len(k_threats) > 0:
            return True
        return False
    
    def legal_castle(self, c: Color) -> dict[CastleSide, bool]:
        legality = copy.deepcopy(self.states.query('castle')[c])
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
                            new_state = self.states.new_state(c = c, move_parameters=('K', k, idx))
                            with self.states.temp(new_state):
                                if self.is_under_check(c):
                                    legality[side] = False
                                    break
                    else:
                        legality[side] = False    
        return legality
        
    def legal_moves(self, c: Color) -> list[tuple[tuple[Piece, int | None, int | None] | str, str | None]]:
        moves, _ = self.prune_blocked(c)
        final = []
        for move_params, move_type in moves:
            new_state = self.states.new_state(move_parameters=move_params, move_type=move_type, c = c)
            with self.states.temp(new_state):
                if not self.is_under_check(c):
                    final.append((move_params, move_type))
        castle = self.legal_castle(c)
        [final.append((side, 'castle')) for side in castle.keys() if castle[side]]
        return final

    def parse_san(self, san: str, c: Color = None, legal_moves = None):
        """Return the move tuple (move_parameters, move_type) for a SAN string."""
        if legal_moves is None:
            legal_moves = self.legal_moves(c)
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
        for (sym, frm, to), move_type in legal_moves:
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
            c: Color,
            legal_moves=None,
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

            if legal_moves is None:
                legal_moves = self.legal_moves(c)

            # --- Disambiguation ---
            # candidates = [
            #     (sym, f, t) for (sym, f, t), mt in legal_moves
            #     if sym == piece and t == to and f != frm
            # ]
            candidates = []
            for mp, mt in legal_moves:
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
            if piece == "P":
                rank = to // 8
                if (c == "w" and rank == 7) or (c == "b" and rank == 0):
                    # Promotion piece is encoded in the move_units
                    for u in self.states.state()["units"]:
                        if u["c"] == c and u["old"] is None and u["new"] == to:
                            san += "=" + u["p"]
                            break

            if move_type == "enpassant":
                san += " ep"

            if check:
                san += "+"
            if mate:
                san += "#"

            return san

    def make_move(self, san: str, c:Color = None, legal_moves = None, narrate = False):
        if legal_moves is None:
            if c is None:
                c = opp(self.states[self.states.pointer]['c'])
            legal_moves = self.legal_moves(c)
        move = self.parse_san(san, legal_moves=legal_moves)
        if move is None:
            print('Invalid san - no corresponding move.')
            return False
        state = self.states.new_state(move_parameters=move[0], move_type=move[1], c = c, san = san, apply = True)
        self.states.apply_state(state, narrate)
        return True
    
    def user_action(self):
        while True:
            action = input('\n--- ACTIONS ---\nM [View Moves]\tU [Undo]\tR [Redo] \tN [New Move]\tQ [Exit Game]\tS [States]\nChoice: ')
            if action in 'MURNQS':
                print('\n')
                break
            else:
                print('Invalid action.')
        return action
        
    def get_san(self):
        return input('Enter move in SAN (Standard Algebraic Notation)): ')

    def is_gameover(self, legal_moves = None):
        c = opp(self.states[self.states.pointer]['c'])
        legal_moves = self.legal_moves(c)
            
        if len(legal_moves) == 0:
            if self.is_under_check(c):
                self.states.end('checkmate', winner = opp(c))
                return True
            else:
                self.states.end('stalemate')
                return True
        return False

    def print_legal_moves(self, c: Color = None,legal_moves = None, raw: bool = False):
        if c is None:
            c = opp(self.states[self.states.pointer]['c'])
        if legal_moves is None:
            legal_moves = self.legal_moves(c)
        if raw:
            [print(mv) for mv in legal_moves]
        else:
            print([self.move_to_san(mv[0], mv[1], c) for mv in legal_moves])
    
    def player_turn(self, narrate = False):
        c = opp(self.states[self.states.pointer]['c'])
        self.pretty_print()
        legal_moves = self.legal_moves(c)        
        if self.is_gameover(legal_moves):
            return
        print(f'\nPly {self.states.pointer}. {formal_color(c)} to move.\n')
        
        while True:
            action = self.user_action()
            if action == "Q":
                return 'quit'
            if action == 'M':
                print([self.move_to_san(mv[0], mv[1], c) for mv in legal_moves])
                # action = self.user_action()
            if action == 'S':
                print(json.dumps(self.states, indent = 4))
                # action = self.user_action()
            if action == 'U':
                self.states.undo()
                print(f"UNDO. {opp(c, True)}'s turn.")
                return
            if action == 'R':
                self.states.redo()
                print(f"REDO. {opp(c, True)}'s turn.")
                return
            if action == 'N':
                break
        while True:
            san = self.get_san()
            move = self.make_move(san, c, legal_moves, narrate)
            if move:
                break
        return

    def print_verdict(self):
        if not self.is_gameover():
            print('Game ongoing')
        else:
            state = self.states.end_state
            s = f"Game ends by {state['result']} on turn {ply_to_turn(state['ply'])}."
            if state['winner'] is not None:
                s += f' {formal_color(state['winner'])} wins.'
            print(s)

    def san_move_list(self):
        s = ''
        for i, ply in enumerate(self.states.keys()):
            if i % 2 != 0:
                s += f'{ply_to_turn(ply)}. {self.states[ply]['san']}'
            else:
                s += f' {self.states[ply]['san']}\n'
        print(s)
        
    def game_loop(self, narrate = False):
        force_q = False
        while self.states.end_state is None and not force_q:
            t = self.player_turn(narrate = narrate)
            if t == 'quit':
                print('\n--- EXITING GAME ---\n')
                force_q = True
            else:
                pass
