import random
import copy
import initial_vars as var

(
    COLORED_PIECES,
    PAWN_DOUBLES,
    MOVEMENTS,
    STARTING_POSITIONS,
    ERROR_MESSAGE,
    COLORS,
    FORMAL,
    LETTERS,
    PIECE_TYPES,
    FORMAL_COLORS,
    INIT_COLOR,
    QSIDE_KSIDE,
    START_BOARD,
    INDEX_MOVES,
    WHITE_PIECES,
    BLACK_PIECES,
) = (
    var.coloredPieces,
    var.dblMoves,
    var.movements,
    var.startingPositions,
    var.error_messages,
    var.colors,
    var.formals,
    var.letters,
    var.piece_types,
    var.formalColors,
    var.initial_color,
    var.startingSideIndices,
    var.starting_bits,
    var.indexMoves,
    var.white_pieces,
    var.black_pieces,
)


def index_filerank(index):
    tup = indexToTuple(index)
    return chr(tup[1] + ord("a")) + str(tup[0] + 1)


def get_newPos(position, move):
    return position + move


def newLine():
    print("\n - - - - - - - - - - - - - - - - - - - - -\n")


def filerank_index(filerank):
    return int(bitIndex((int(filerank[1]) - 1, ord(filerank[0]) - ord("a"))))


def previousMovesPrinted(turnMoves):
    (print(turnmove) for turnmove in turnMoves)


def bit_init():
    return START_BOARD.copy()


def bitIndex(tuple):
    index = tuple[0] * 8 + tuple[1]
    return index


def indexToTuple(index):
    row = index // 8
    col = index % 8
    return row, col


def indexToPos(bin):
    indices = []
    for i in range(64):
        if bin[i] == "1":
            indices.append(i)
    return indices


def changeBit(b, i, bool):
    if bool == True:
        b = b[:i] + "1" + b[i + 1 :]
    elif bool == False:
        b = b[:i] + "0" + b[i + 1 :]
    return b


def onBoard(index):
    if ((index // 8) in range(8)) and ((index % 8) in range(8)):
        return True
    return False


def pieceInIndex(board, index):
    for piece, binary in board.items():
        if (binary[index]) == "1":
            return piece
    return None


def bit_init():
    return START_BOARD.copy()


def genEdges():
    edges = []
    for i in range(8):
        for j in range(8):
            if (i == 0) or (i == 7):
                edges.append((i, j))
            elif (j == 0) or (j == 7):
                edges.append((i, j))
    return edges


def gameConfig():
    inp = input("Show available Moves by default? Y/N\n")
    disp = input("(Recommended) Display Board after move? Y/N\n")
    prev = input("Display Previous Moves By Default? Y/N\n")
    if (inp == "y") or (inp == "Y"):
        defaultMoveShow = True
    else:
        defaultMoveShow = False
    if (disp == "y") or (disp == "Y"):
        defaultDispShow = True
    else:
        defaultDispShow = False
    if (prev == "y") or (prev == "Y"):
        defaultTurnsShow = True
    else:
        defaultTurnsShow = False
    return defaultMoveShow, defaultDispShow, defaultTurnsShow


class Board:
    def __init__(self, config, board=None):
        if board == None:
            self.board = bit_init()
        else:
            self.board = board
        self.c = "w"
        self.king = self.setPiece("King")
        self.queen = self.setPiece("Queen")
        self.pawn = self.setPiece("Pawn")
        self.rook = self.setPiece("Rook")
        self.bishop = self.setPiece("Bishop")
        self.knight = self.setPiece("Knight")
        self.movements = INDEX_MOVES
        self.config = config
        self.kingMoved = {"w": False, "b": False}
        self.rookQSide = {"w": False, "b": False}
        self.rookKSide = {"w": False, "b": False}
        self.turnMoves = []
        self.turnNumber = 1
        self.dblMoveIndex = None
        self.whiteOnTop = False
        self.Break = False
        self.running = True
        self.active = self.colorBoards()[0]
        self.inactive = self.colorBoards()[1]
        self.edges = genEdges()
        self.noDoubles = {"w": False, "b": False}
        self.pawnDoubles = self.pawn_starting_double()
        self.check = False
        self.stalemate = False
        self.checkmate = False
        self.move = None
        self.kingCastle, self.queenCastle = False, False
        self.all = self.allMoves()
        self.inp = None

    def setPiece(self, piece):
        return COLORED_PIECES[self.c][piece]

    def alternateColor(color):
        if color == COLORS[0]:
            return COLORS[1]

        elif color == COLORS[1]:
            return COLORS[0]

    def colorBoards(self):
        active, inactive = {}, {}
        for piece in self.board:
            if self.c == "w":
                if piece in WHITE_PIECES:
                    active[piece] = self.board[piece]
                else:
                    inactive[piece] = self.board[piece]
            else:
                if piece in BLACK_PIECES:
                    active[piece] = self.board[piece]
                else:
                    inactive[piece] = self.board[piece]
        return active, inactive

    def identifyKing(self):
        if self.c == "w":
            return "K"
        else:
            return "k"

    def activePawn(self):
        if self.c == "w":
            return "P"
        else:
            return "p"

    def pawn_starting_double(self):
        doublesAvailable = []
        for i in range(64):
            if (self.board[self.pawn][i] == "1") & (START_BOARD[self.pawn][i] == "1"):
                doublesAvailable.append(i)
        if len(doublesAvailable) == 0:
            self.noDoubles[self.c] = True
        return doublesAvailable

    def isCaptured(self, move):
        for opp in self.inactive:
            if self.inactive[opp][move] == "1":
                return opp
        return False

    def display(self):
        empty = "-"
        string = ""
        string += "   | "
        sepLine = ""
        sepLine += "    "
        sepLine += " _  " * 8
        for i in range(7):
            string += LETTERS[i]
            string += " | "
        string += LETTERS[7]
        squares = []
        for piece in self.board:
            for pos in range(64):
                if self.board[piece][pos] == "1":
                    squares.append((piece, pos))
        posns = []
        for i in range(64):
            isin = False
            for p in squares:
                if p[1] == i:
                    posns.append(p[0])
                    isin = True
                    break
            if not isin:
                posns.append(empty)
        if self.whiteOnTop:
            print(string)
            print(sepLine)
            for row in range(8):
                r = (
                    f"{row + 1} || {posns[8*row + 0]}"
                    + " | "
                    + f"{posns[8*row + 1]}"
                    + " | "
                    + f"{posns[8*row + 2]}"
                    + " | "
                    + f"{posns[8*row + 3]}"
                    + " | "
                    + f"{posns[8*row + 4]}"
                    + " | "
                    + f"{posns[8*row + 5]}"
                    + " | "
                    + f"{posns[8*row + 6]}"
                    + " | "
                    + f"{posns[8*row + 7]}"
                )
                print(r)
        elif not self.whiteOnTop:
            for row in range(7, -1, -1):
                r = (
                    f"{row + 1} || {posns[8*row + 0]}"
                    + " | "
                    + f"{posns[8*row + 1]}"
                    + " | "
                    + f"{posns[8*row + 2]}"
                    + " | "
                    + f"{posns[8*row + 3]}"
                    + " | "
                    + f"{posns[8*row + 4]}"
                    + " | "
                    + f"{posns[8*row + 5]}"
                    + " | "
                    + f"{posns[8*row + 6]}"
                    + " | "
                    + f"{posns[8*row + 7]}"
                )
                print(r)
            print(sepLine)
            print(string)

    def preMove(self):
        print(f"Turn {self.turnNumber} | {FORMAL_COLORS[self.c]} to move")
        if self.config[0]:
            self.printAvailableMoves()
        if self.config[1]:
            self.display()
        if self.config[2]:
            previousMovesPrinted(self.turnMoves)
        moveReady = False
        while not moveReady:
            self.takeAction()
            if self.input == "end":
                self.running, self.Break = False, True
                break
            move = self.file_rank_compile(self.input)
            if move != None:
                self.move = move
                moveReady = True
                break

    def qualifyPieceInput(self, act):
        if len(act) == 2:
            return self.pawn, filerank_index(act.lower())
        else:
            if self.c == "w":
                p = act[0].upper()
            else:
                p = act[0].lower()
            return p, filerank_index((act[1].lower() + act[2]))

    def file_rank_compile(self, act):
        if (act == "O-O") and (act in self.all):
            return "O-O"
        elif (act == "O-O-O") and (act in self.all):
            return "O-O-O"
        else:
            piece, mv = self.qualifyPieceInput(act)
            if self.c == "w":
                p = piece.upper()
            elif self.c == "b":
                p = piece.lower()
            for i in range(2):
                for move in self.all[i]:
                    if (mv == move[1][1]) and (move[0] == p):
                        return move
            return None

    def printAvailableMoves(self):
        pass
        allMoves = self.all
        mvs = []
        for i in range(2):
            for move in allMoves[i]:
                mvs.append(
                    str(str(FORMAL[move[0]]) + " to " + str(index_filerank(move[1][1])))
                )
        string = ""
        for Move in mvs[:-1]:
            string += Move
            string += " | "
        string += mvs[-1]
        return string

    def printPrevMoves(self):
        (print(turnmove) for turnmove in self.turnMoves)

    def takeAction(self):
        inp = input(
            "- - - - - GameOptions - - - - - \n|| Previous Moves [P] || Available Moves [A] || Stop Game [S] ||\n|| Make a Piece-Square Move [O-O : King-Side Castle | O-O-O : Queen-side Castle] ||\n|| Action ||>> "
        )
        newLine()
        if inp == "P":
            self.printPrevMoves()
            newLine()
            return
        elif inp == "A":
            self.printAvailableMoves()
            newLine()
            return
        elif inp == "S":
            self.input = "end"
        elif inp == "D":
            self.display()
            return
        else:
            self.input = inp

    def allMoves(self):
        """
        Returns 2 lists: Non-capture moves, and Capture moves for current board conditions.

        """
        Moves = [], []
        for piece in self.active:
            moves = self.postCheckMoves(piece)
            for i in range(2):
                for m in moves[i]:
                    Moves[i].append((piece, m))
        if self.queenCastle:
            moves[0].append("O-O-O")
        if self.kingCastle:
            moves[0].append("O-O")
        return Moves

    def boardAfterMove(self, piece, move):
        # move is one of the tuples in lists provided by
        newBoard = copy.deepcopy(self.board)
        if len(move) == 3:
            victim = pieceInIndex(move[2], newBoard)
            newBoard[piece] == changeBit(newBoard[piece], move[0], False)
            newBoard[piece] == changeBit(newBoard[piece], move[1], True)
            newBoard[victim] == changeBit(newBoard[victim], move[2], False)
        else:
            victim = self.isCaptured(move[1])
            newBoard[piece] = changeBit(newBoard[piece], move[0], False)
            newBoard[piece] = changeBit(newBoard[piece], move[1], True)
            if victim != False:
                newBoard[victim] = changeBit(newBoard[victim], move[1], False)
        return newBoard

    def postCheckMoves(self, piece):
        legal1, capture1 = self.moves_postBlock(piece)
        legal2, capture2 = [], []
        for move in legal1:
            newboard = self.boardAfterMove(piece, move)
            check = self.isCheck(boardAfterMove=newboard)
            if not check:
                legal2.append(move)
        for move in capture1:
            newboard = self.boardAfterMove(piece, move)
            check = self.isCheck(boardAfterMove=newboard)
            if not check:
                capture2.append(move)
        return legal2, capture2

    def boardSide(self, piece, side):
        string = ""
        if side == "q":
            for i in range(64):
                if i % 8 == 0:
                    string += self.board[piece][i]
        elif side == "k":
            for i in range(1, 65):
                if i % 8 == 0:
                    string += self.board[piece][i - 1]
        return string

    def pieceHasMoved(self, piece, side=None):
        startBoard = START_BOARD
        if self.c == "w":
            piece = piece.upper()
        if side != None:
            if self.boardSide(piece, side) != self.boardSide(piece, side):
                return True
        if self.board[piece] != startBoard[piece]:
            return True
        return False

    def bit_to_indices(self, piece):
        indices = []
        for i in range(64):
            if self.board[piece][i] == "1":
                indices.append(i)
        return indices

    def isMate(self):
        moves = self.allMoves()
        for i in range(2):
            if len(moves[0]) != 0:
                return
        if self.isCheck():
            return "checkmate"
        else:
            return "stalemate"

    def boardTurn(self):
        self.affectMove()
        if self.isMate() == "checkmate":
            self.checkmate == True
            self.running = False
            pass
        elif self.isMate() == "stalemate":
            self.stalemate = True
            self.running = False
            pass

    def moves_postBlock(self, piece, boardAfterMove=None):
        """
        returns two lists of tuples, where the tuples are (old_position, new_position).
        1st list: non-capture moves
        2nd list: capture moves

        params::
        piece: string, e.g. 'Q' or 'q', as in startingPositions & MOVEMENTS (Uppercase = White, Lowercase = Black)
        boardState: the boardState dict of {piece: binary number} (binary represents position(s))
        MOVEMENTS:  dictionary containing piece movement rules
        audit: boolean; set to True to assess step-by-step evaluations
        """
        if boardAfterMove != None:
            board = boardAfterMove
        else:
            board = self.board
        edges = self.edges
        all_positions = self.bit_to_indices(piece)
        legal, captures = [], []
        for position in all_positions:
            if piece == self.pawn:
                dbl = False
                fwd = []
                forwardSquare = self.movements[piece][0][0]
                captureSquares = self.movements[piece][1]
                if self.dblMoveIndex != None:
                    if position not in edges:
                        if (position + 1 == self.dblMoveIndex) or (
                            position - 1 == self.dblMoveIndex
                        ):
                            if piece == "P":
                                captures.append(
                                    (position, self.dblMoveIndex + 8, self.dblMoveIndex)
                                )
                            if piece == "p":
                                captures.append(
                                    (position, self.dblMoveIndex - 8, self.dblMoveIndex)
                                )
                    else:
                        for i in [1, -1]:
                            if onBoard(position + i):
                                if position + i == self.dblMoveIndex:
                                    if piece == "P":
                                        captures.append(
                                            (
                                                position,
                                                self.dblMoveIndex + 8,
                                                self.dblMoveIndex,
                                            )
                                        )
                                    if piece == "p":
                                        captures.append(
                                            (
                                                position,
                                                self.dblMoveIndex - 8,
                                                self.dblMoveIndex,
                                            )
                                        )

                if position in self.pawnDoubles:
                    dbl = True
                    if piece == "p":
                        fwd.append(get_newPos(position, -16))
                    elif piece == "P":
                        fwd.append(get_newPos(position, 16))
                fwd.append(get_newPos(position, forwardSquare))
                cap1, cap2 = get_newPos(position, captureSquares[0]), get_newPos(
                    position, captureSquares[1]
                )
                if pieceInIndex(board, fwd[0]) == None:
                    if dbl:
                        if pieceInIndex(board, fwd[1]) == None:
                            legal.append((position, fwd[1]))
                    legal.append((position, fwd[0]))
                if pieceInIndex(board, cap1) in self.inactive:
                    captures.append((position, cap1))
                if pieceInIndex(board, cap2) in self.inactive:
                    captures.append((position, cap2))
            else:
                for move_sect in self.movements[piece]:
                    block, capture, block, edge, offBoard = (
                        False,
                        False,
                        False,
                        False,
                        False,
                    )
                    for i, move in enumerate(move_sect):
                        newPos = get_newPos(position, move)
                        if onBoard(newPos):
                            squareCheck = pieceInIndex(board, newPos)
                            if squareCheck != None:
                                if squareCheck in self.inactive:
                                    [
                                        legal.append(
                                            (position, get_newPos(position, mv))
                                        )
                                        for mv in move_sect[:i]
                                    ]
                                    captures.append((position, newPos))
                                    capture = True
                                    break
                                elif squareCheck in self.active:
                                    [
                                        legal.append(
                                            (position, get_newPos(position, mv))
                                        )
                                        for mv in move_sect[:i]
                                    ]
                                    block = True
                                    break
                            if newPos in edges:
                                [
                                    legal.append((position, get_newPos(position, mv)))
                                    for mv in move_sect[: i + 1]
                                ]
                                edge = True
                                break
                        else:
                            [
                                legal.append((position, get_newPos(position, mv)))
                                for mv in move_sect[:i]
                            ]
                            offBoard = True
                            break
                    if (not block) and (not capture) and (not edge) and (not offBoard):
                        [
                            legal.append((position, get_newPos(position, mv)))
                            for mv in move_sect
                        ]
        return legal, captures

    def isThreatened(self, piece, boardAfterMove=None):
        threats = []
        if boardAfterMove != None:
            for opp in self.inactive:
                captures = self.moves_postBlock(opp, boardAfterMove)[1]
                caps = len(captures)
                if caps > 0:
                    for i in range(caps):
                        if boardAfterMove[piece][captures[i][1]] == "1":
                            threats.append(captures[i])
            return threats
        else:
            for opp in self.inactive:
                captures = self.moves_postBlock(opp, boardAfterMove)[1]
                caps = len(captures)
                if caps > 0:
                    for i in range(caps):
                        if self.active[piece][captures[i][1]] == "1":
                            threats.append(captures[i])
            return threats

    def isCheck(self, boardAfterMove=None):
        if boardAfterMove != None:
            if len(self.isThreatened(self.king, boardAfterMove)) > 0:
                return True
            else:
                return False
        else:
            if len(self.isThreatened(self.king)) > 0:
                self.check = True
                return f"{FORMAL[self.c]} in check", self.isThreatened(self.king)
            else:
                return False

    def afterTurn(self):
        self.c = self.alternateColor()
        self.active = self.colorBoards()[0]
        self.inactive = self.colorBoards()[1]
        self.king = self.setPiece("King")
        self.queen = self.setPiece("Queen")
        self.pawn = self.setPiece("Pawn")
        self.rook = self.setPiece("Rook")
        self.bishop = self.setPiece("Bishop")
        self.knight = self.setPiece("Knight")
        self.allMoves = self.allMoves()
        if not self.noDoubles[self.c]:
            self.pawnDoubles = self.pawn_starting_double()

    def getIndexBySide(self, piece, side):
        return QSIDE_KSIDE[self.c][side][piece]

    def castlePossible(self):
        if self.kingMoved[self.c]:
            return False, False
        if self.isCheck():
            return False, False
        kingQueen = True, True
        if self.rookQSide[self.c]:
            kingQueen[1] = False
        if self.rookKSide[self.c]:
            kingQueen[0] = True
        else:
            if not self.castlePathClear("q"):
                kingQueen[1] = False
            if not self.castlePathClear("k"):
                kingQueen[0] = False
        self.kingCastle = kingQueen[0]
        self.queenCastle = kingQueen[1]

    def castlePathClear(self, side):
        for piece in [self.bishop, self.knight]:
            if pieceInIndex(self.board, self.getIndexBySide(piece, side)) != None:
                return False
            if self.checkDummyKing(self.getIndexBySide(piece, side)):
                return False
        return True

    def checkDummyKing(self, index):
        boardCheck = self.board.copy()
        boardCheck[self.king] = 64 * "0"
        boardCheck[self.king] = changeBit(boardCheck[self.king], index, True)
        if isCheck(self.active, boardCheck, self.dblMoveIndex):
            return False
        return True

    def affectMove(self):
        move = self.move
        self.dblMoveIndex = self.pawnDoubleMove_index(move)
        if move == "O-O":
            self.castleMove("k")
        elif move == "O-O-O":
            self.castleMove("q")
        else:
            allMoves = self.all
            for section in allMoves:
                for mv in section:
                    if mv[1] == move:
                        self.board = self.boardAfterMove(move[0], move[1])

    def pawnDoubleMove_index(self, move):
        if (self.pawn, move) in self.pawnDoubles:
            return move[1]
        return None

    def castleMove(self, side):
        self.board[self.king] = changeBit(
            self.board[self.king], QSIDE_KSIDE[self.c][side][self.king], False
        )
        self.board[self.rook] = changeBit(
            self.board[self.rook], QSIDE_KSIDE[self.c][side][self.rook], False
        )
        if side == "k":
            self.board[self.king] = changeBit(
                self.board[self.king], QSIDE_KSIDE[self.c][side][self.king] + 2, False
            )
            self.board[self.rook] = changeBit(
                self.board[self.rook], QSIDE_KSIDE[self.c][side][self.rook] - 2, False
            )
        if side == "q":
            self.board[self.king] = changeBit(
                self.board[self.king], QSIDE_KSIDE[self.c][side][self.king] - 2, False
            )
            self.board[self.rook] = changeBit(
                self.board[self.rook], QSIDE_KSIDE[self.c][side][self.rook] + 3, False
            )

    def afterBoard(self):
        if self.running:
            if not self.Break:
                self.display()
                if self.stalemate:
                    print(f"Game Over (Stalemate)")
                elif self.checkmate:
                    print(f"Game Over. {FORMAL_COLORS[self.c]} Wins by Checkmate!")
                playagain = input("Play Again [Y] | Exit [Any Key] \n>> ")
                if playagain == "Y":
                    return True
            else:
                self.__del__()
                return False

    def __del__(self):
        print("See You Next Time!")
