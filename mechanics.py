import copy, initial_vars as var

(
    PIECE_VALUES,
    OPPOSITE_EDGES,
    EDGES,
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
    var.pieceValues,
    var.oppositeEdges,
    var.edges,
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


def kingLocation(board, piece):
    for i in range(64):
        if board[piece][i] == "1":
            return i


def boardValue(board):
    king = ["k", "K"]
    value = 0
    for piece in board:
        if piece not in king:
            for i in bit_to_indices(piece, board):
                value += PIECE_VALUES[piece]
    return value


def valueAfterMove(board, move, inactive):
    piece, mv = move
    newboard = boardAfterMove(board, piece, mv, inactive)
    value = boardValue(newboard)
    return value, mv


def getPieceEvaluation(piece, board, dblMoveIndex, active, inactive):
    """
    Output Elements:
    0 - piece (str) ||
    1 - threatenedBy (list) ||
    2 - protectedB (list) ||
    3 - isProtecting (list) ||
    4 - canCapture (list) ||
    5 - canMove (list) ||

    --> lists: elements are tuples of 2 - piece & move-tuple (old Index, new Index)
    """
    output = [], [], [], [], []
    canMove, canCapture, isProtecting = beforeCheckMoves(
        piece, board, dblMoveIndex, active, inactive
    )
    protectedBy, threatenedBy = isProtected(
        piece, board, dblMoveIndex, active, inactive
    ), isThreatened(piece, board, dblMoveIndex, active, inactive)
    output = [piece]
    for li in [threatenedBy, protectedBy, isProtecting, canCapture, canMove]:
        outputEle = []
        for mv in li:
            outputEle.append((pieceInIndex(board, mv[1]), mv))
        output.append(outputEle)
    return output


def protectedBy(piece, board):
    color = getColor(piece=piece)
    active, inactive = colorBoards(board, color)
    all = all
    for piece in active:
        allProtections = []


def getActive(board, piece):
    active = {}
    if piece in WHITE_PIECES:
        for piece in WHITE_PIECES:
            active[piece] = board[piece]
    else:
        for piece in BLACK_PIECES:
            active[piece] = board[piece]
    return active


def index_filerank(index):
    tup = indexToTuple(index)
    return chr(tup[1] + ord("a")) + str(tup[0] + 1)
    

def getFilerankMove(piece, move):
    filerank = index_filerank(move[1])
    if piece in ['P', 'p']:
        return filerank
    return str(piece) + str(filerank)


def get_newPos(position, move):
    return position + move


def newLine():
    print("\n - - - - - - - - - - - - - - - - - - - - -\n")


def filerank_index(filerank):
    return int(bitIndex((int(filerank[1]) - 1, ord(filerank[0]) - ord("a"))))


def previousMovesPrinted(turnMoves):
    (print(turnmove) for turnmove in turnMoves)


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


def bit_to_indices(piece, board):
    indices = []
    for i in range(64):
        if board[piece][i] == "1":
            indices.append(i)
    return indices


def getColor(active=None, piece=None):
    if active != None:
        if list(active.keys())[0] in WHITE_PIECES:
            return "w"
        else:
            return "b"
    elif piece != None:
        if piece.isupper():
            return "w"
        else:
            return "b"


def afterCheckMoves(piece, board, dblMoveIndex, active, inactive):
    legal, captures = beforeCheckMoves(piece, board, dblMoveIndex, active, inactive)[:2]
    noncheck = []
    color = getColor(piece = piece)
    for movesList in [legal, captures]:
            nonchecksEle = []
            for move in movesList:
                if not disallowedCheck(color, piece, board, move, inactive):
                    nonchecksEle.append(move)
            noncheck.append(nonchecksEle)
    return noncheck


def disallowedCheck(color, piece, board, move, inactive):
    newboard = boardAfterMove(board, piece, move, inactive)
    newActive, newInactive = colorBoards(newboard, color)
    check = isCheck(newboard, None, newActive, newInactive)
    return check


def pawn_starting_double(board, pawn):
    doublesAvailable = []
    for i in range(64):
        if (board[pawn][i] == "1") & (START_BOARD[pawn][i] == "1"):
            doublesAvailable.append(i)
    return doublesAvailable


def isThreatened(piece, board, dblMoveIndex, active, inactive):
    threats = []
    for opp in inactive:
        captures = beforeCheckMoves(opp, board, dblMoveIndex, inactive, active)[1]
        if len(captures) > 0:
            for cap in captures:
                if active[piece][cap[1]] == "1":
                    threats.append(cap)
    return threats


def isProtected(piece, board, dblMoveIndex, active, inactive):
    protections = []
    for ally in active:
        protects = beforeCheckMoves(ally, board, dblMoveIndex, active, inactive)[2]
        if len(protects) > 0:
            for prot in protects:
                if active[piece][prot[1]] == "1":
                    protections.append(prot)
    return protections


def isCheck(board, dblMoveIndex, active, inactive):
    kingItem = ["K", "k"]
    for item in kingItem:
        if item in active:
            activeKing = item
    if len(isThreatened(activeKing, board, dblMoveIndex, active, inactive)) > 0:
        return True
    else:
        return False


def isCaptured(move, inactive):
    for opp in inactive:
        if inactive[opp][move] == "1":
            return opp
    return False


def colorBoards(board, color):
    active, inactive = {}, {}
    for piece in board:
        if color == "w":
            if piece in WHITE_PIECES:
                active[piece] = board[piece]
            else:
                inactive[piece] = board[piece]
        else:
            if piece in BLACK_PIECES:
                active[piece] = board[piece]
            else:
                inactive[piece] = board[piece]
    return active, inactive


class Board:
    def __init__(self, config, board=None):
        if board == None:
            self.board = bit_init()
        else:
            self.board = board
        self.c = INIT_COLOR
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
        self.active, self.inactive = colorBoards(self.board, self.c)
        self.edges = EDGES
        self.noDoubles = {"w": False, "b": False}
        self.pawnDoubles = pawn_starting_double(self.board, self.pawn)
        self.check = False
        self.stalemate = False
        self.checkmate = False
        self.move = None
        self.kingCastle = False
        self.queenCastle = False
        self.all = self.allMoves()
        self.input = None
        self.endAll = False
        self.netPieceValue = 0

    def setPiece(self, piece):
        return COLORED_PIECES[self.c][piece]

    def alternateColor(self):
        if self.c == COLORS[0]:
            return COLORS[1]
        elif self.c == COLORS[1]:
            return COLORS[0]

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
            if self.input == "S":
                self.running, self.Break = False, True
                break
            move = self.file_rank_compile(self.input)
            if move != None:
                self.move = move
                moveReady = True
                break
            else:
                print("Not a Valid Move.")
                newLine()

    def qualifyPieceInput(self, act):
        if len(act) == 1:
            pass
        elif len(act) == 2:
            return self.pawn, filerank_index(act.lower())
        else:
            if self.c == "w":
                p = act[0].upper()
            else:
                p = act[0].lower()
            return p, filerank_index((act[1].lower() + act[2]))

    def file_rank_compile(self, act):
        if len(act) != 1:
            if (act == "O-O") and (act in self.all[0]):
                return "O-O"
            elif (act == "O-O-O") and (act in self.all[0]):
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
        allMoves = self.all
        if (len(self.all[0]) == 0) and (len(self.all[1]) == 0):
            print("|| No available Moves || ")
            return None
        mvs = []
        for i in range(2):
            for move in allMoves[i]:
                if (move == "O-O") or (move == "O-O-O"):
                    mvs.append(move)
                else:
                    mvs.append(
                        str(
                            str(FORMAL[move[0]])
                            + " to "
                            + str(index_filerank(move[1][1]))
                        )
                    )
        string = ""
        for Move in mvs[:-1]:
            string += Move
            string += " | "
        string += mvs[-1]
        print(string)

    def printPrevMoves(self):
        (print(turnmove) for turnmove in self.turnMoves)

    def takeAction(self):
        inp = input(
            "- - - - - GameOptions - - - - - \n|| Previous Moves [P] || Available Moves [A] || Stop Game [S] ||\n|| Make a Piece-Square Move [O-O : King-Side Castle | O-O-O : Queen-side Castle] ||\n|| Action ||>> "
        )
        newLine()
        self.input = inp
        if inp == "P":
            self.printPrevMoves()
            newLine()
            return
        elif inp == "A":
            self.printAvailableMoves()
            newLine()
            return
        elif inp == "S":
            self.input = "S"
        elif inp == "D":
            self.display()
            return

    def allMoves(self):
        """
        Returns 2 lists: Non-capture moves, and Capture moves for current board conditions.

        """
        Moves = [], []
        for piece in self.active:
            moves = afterCheckMoves(
                piece, self.board, self.dblMoveIndex, self.active, self.inactive
            )
            for i in range(2):
                for m in moves[i]:
                    Moves[i].append((piece, m))
        if self.queenCastle:
            Moves[0].append("O-O-O")
        if self.kingCastle:
            Moves[0].append("O-O")
        return Moves

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

    def boardTurn(self):
        self.affectMove()

    def afterTurn(self):
        self.promotion()
        self.turnNumber = self.turnIncreaser()
        self.c = self.alternateColor()
        self.active, self.inactive = colorBoards(self.board, self.c)
        self.king = self.setPiece("King")
        self.queen = self.setPiece("Queen")
        self.pawn = self.setPiece("Pawn")
        self.rook = self.setPiece("Rook")
        self.bishop = self.setPiece("Bishop")
        self.knight = self.setPiece("Knight")
        if not self.noDoubles[self.c]:
            self.pawnDoubles = pawn_starting_double(self.board, self.pawn)
        self.kingCastle, self.queenCastle = self.castlePossible()
        self.check = isCheck(self.board, self.dblMoveIndex, self.active, self.inactive)
        self.all = self.allMoves()
        if (len(self.all[0]) == 0) and (len(self.all[1]) == 0):
            if self.check:
                self.checkmate = True
                self.running = False
            else:
                self.stalemate = True
                self.running = False
        else:
            self.checkmate, self.stalemate = False, False
        self.netPieceValue = boardValue(self.board)

    def getIndexBySide(self, piece, side):
        return QSIDE_KSIDE[self.c][side][piece]

    def castlePossible(self):
        if self.kingMoved[self.c]:
            return False, False
        if isCheck(self.board, self.dblMoveIndex, self.active, self.inactive):
            return False, False
        king, queen = True, True
        if self.rookQSide[self.c]:
            queen = False
        if self.rookKSide[self.c]:
            king = False
        else:
            if not self.castlePathClear("q"):
                queen = False
            if not self.castlePathClear("k"):
                king = False
        return king, queen

    def castlePathClear(self, side):
        if side == "q":
            for piece in [self.queen, self.bishop, self.knight]:
                if pieceInIndex(self.board, self.getIndexBySide(piece, side)) != None:
                    return False
            for piece in [self.queen, self.bishop]:
                if self.checkDummyKing(self.getIndexBySide(piece, side)):
                    return False
        else:
            for piece in [self.bishop, self.knight]:
                if pieceInIndex(self.board, self.getIndexBySide(piece, side)) != None:
                    return False
                if not self.checkDummyKing(self.getIndexBySide(piece, side)):
                    return False
        return True

    def checkDummyKing(self, index):
        boardCheck = self.board.copy()
        boardCheck[self.king] = 64 * "0"
        boardCheck[self.king] = changeBit(boardCheck[self.king], index, True)
        if isCheck(boardCheck, None, self.active, self.inactive):
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
                    if mv == move:
                        self.board = boardAfterMove(
                            self.board, move[0], move[1], self.inactive
                        )

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
                self.board[self.king], QSIDE_KSIDE[self.c][side][self.king] + 2, True
            )
            self.board[self.rook] = changeBit(
                self.board[self.rook], QSIDE_KSIDE[self.c][side][self.rook] - 2, True
            )
        if side == "q":
            self.board[self.king] = changeBit(
                self.board[self.king], QSIDE_KSIDE[self.c][side][self.king] - 2, True
            )
            self.board[self.rook] = changeBit(
                self.board[self.rook], QSIDE_KSIDE[self.c][side][self.rook] + 3, True
            )

    def afterBoard(self):
        if not self.running:
            if not self.Break:
                self.display()
                if self.stalemate:
                    print(f"Game Over (Stalemate)")
                elif self.checkmate:
                    print(
                        f"Game Over. {FORMAL_COLORS[self.alternateColor()]} Wins by Checkmate!"
                    )
                playagain = input("Play Again [Y] | Exit [Any Key] \n>> ")
                if playagain == "Y":
                    return True
                else:
                    self.endAll = True
            elif self.Break:
                print(f"Game Stopped.")
                playagain = input("Play Again [Y] | Exit [Any Key] \n>> ")
                if playagain == "Y":
                    newLine()
                    return True
                else:
                    self.endAll = True

    def __del__(self):
        print(f"{'-'*35}")

    def manualMove(self, filerank):
        self.move = self.file_rank_compile(filerank)
        self.affectMove()
        if self.checkmate:
            self.checkmate == True
            self.running = False
            pass
        elif self.stalemate:
            self.stalemate = True
            self.running = False
            pass

    def turnIncreaser(self):
        if self.c == "b":
            return self.turnNumber + 1
        else:
            return self.turnNumber

    def promotion(self):
        ops = self.pawnOnOppositeEdge()
        if len(ops) != 0:
            self.board[self.pawn] = changeBit(self.board[self.pawn], ops[0], False)
            self.board[self.queen] = changeBit(self.board[self.queen], ops[0], True)

    def pawnOnOppositeEdge(self):
        pawnOpposites = []
        pawnPositions = bit_to_indices(self.pawn, self.board)
        oppositeEdge = OPPOSITE_EDGES[self.c]
        for i in pawnPositions:
            if i in oppositeEdge:
                pawnOpposites.append(i)
        return pawnOpposites

    @classmethod
    def projectBoard(self):
        pass
        

def boardAfterMove(board, piece, move, inactive):
    # move is one of the tuples in lists provided by
    newBoard = board.copy()
    if len(move) == 3:
        victim = pieceInIndex(move[2], newBoard)
        newBoard[piece] == changeBit(newBoard[piece], move[0], False)
        newBoard[piece] == changeBit(newBoard[piece], move[1], True)
        newBoard[victim] == changeBit(newBoard[victim], move[2], False)
    else:
        victim = isCaptured(move[1], inactive)
        newBoard[piece] = changeBit(newBoard[piece], move[0], False)
        newBoard[piece] = changeBit(newBoard[piece], move[1], True)
        if victim != False:
            newBoard[victim] = changeBit(newBoard[victim], move[1], False)
    return newBoard


def beforeCheckMoves(piece, board, dblMoveIndex, active, inactive):
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
    edges = EDGES
    all_positions = bit_to_indices(piece, board)
    legal, captures, protections = [], [], []
    color = getColor(active)
    for position in all_positions:
        if piece in ["p", "P"]:
            pawnDoubles = pawn_starting_double(board, piece)
            dbl = False
            fwd = []
            forwardSquare = INDEX_MOVES[piece][0][0]
            captureSquares = INDEX_MOVES[piece][1]
            if dblMoveIndex != None:
                if position not in edges:
                    if (position + 1 == dblMoveIndex) or (position - 1 == dblMoveIndex):
                        if piece == "P":
                            captures.append((position, dblMoveIndex + 8, dblMoveIndex))
                        if piece == "p":
                            captures.append((position, dblMoveIndex - 8, dblMoveIndex))
                else:
                    for i in [1, -1]:
                        if onBoard(position + i):
                            if position + i == dblMoveIndex:
                                if piece == "P":
                                    captures.append(
                                        (
                                            position,
                                            dblMoveIndex + 8,
                                            dblMoveIndex,
                                        )
                                    )
                                if piece == "p":
                                    captures.append(
                                        (
                                            position,
                                            dblMoveIndex - 8,
                                            dblMoveIndex,
                                        )
                                    )

            if position in pawnDoubles:
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
            for cap in [cap1, cap2]:
                p = pieceInIndex(board, cap)
                if p in inactive:
                    captures.append((position, cap))
                elif p in active:
                    protections.append((position, cap))
        else:
            for move_sect in INDEX_MOVES[piece]:
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
                            if squareCheck in inactive:
                                [
                                    legal.append((position, get_newPos(position, mv)))
                                    for mv in move_sect[:i]
                                ]
                                captures.append((position, newPos))
                                capture = True
                                break
                            elif squareCheck in active:
                                [
                                    legal.append((position, get_newPos(position, mv)))
                                    for mv in move_sect[:i]
                                ]
                                protections.append((position, newPos))
                                block = True
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
    return legal, captures, protections
       