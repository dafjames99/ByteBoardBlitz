import copy
import initial_vars as var

PAWN_DOUBLES, MOVEMENTS, STARTING_POSITIONS, ERROR_MESSAGE, COLORS, FORMAL = (
    var.dblMoves,
    var.movements,
    var.startingPositions,
    var.error_messages,
    var.colors,
    var.formals,
)
LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]


# Tuple/index/Bit operations
def bitIndex(tuple):
    index = tuple[0] * 8 + tuple[1]
    return index


def indexToTuple(index):
    row = index // 8
    col = index % 8
    return row, col


def changeBit(b, i, bool):
    if bool == True:
        b = b[:i] + "1" + b[i + 1 :]
    elif bool == False:
        b = b[:i] + "0" + b[i + 1 :]
    return b


def indexToPos(bin):
    tups = []
    for i in range(64):
        if bin[i] == "1":
            tups.append(indexToTuple(i))
    return tups


# create & comprehend board
def bit_initialise(positions):
    board = {}
    for item, posns in positions.items():
        binary = "0" * 64
        for position in posns:
            binary = changeBit(binary, bitIndex(position), True)
        board[item] = binary
    return board


# comprehend coloured boards
def colourDict(boardState):
    dict_white = {}
    dict_black = {}
    for piece in boardState:
        if piece in ["K", "Q", "R", "B", "N", "P"]:
            dict_white[piece] = boardState[piece]
        elif piece in ["k", "q", "r", "b", "n", "p"]:
            dict_black[piece] = boardState[piece]
    return dict_white, dict_black


def setActive(piece, boardState):
    w, b = colourDict(boardState)
    if piece in w:
        active, inactive = w, b
    else:
        active, inactive = b, w
    return active, inactive


def pieceInIndex(index, dict):
    for piece in dict:
        if (dict[piece][index]) == "1":
            return piece
    return None


def setActiveByColor(color, boardState):
    w, b = colourDict(boardState)
    if color == "w":
        return w
    elif color == "b":
        return b


def get_newPos(position, move):
    return (position[0] + move[0], position[1] + move[1])


def genEdges():
    edges = []
    for i in range(8):
        for j in range(8):
            if (i == 0) or (i == 7):
                edges.append((i, j))
            elif (j == 0) or (j == 7):
                edges.append((i, j))
    return edges


def onBoard(pos):
    if (pos[0] in range(8)) and (pos[1] in range(8)):
        return True
    return False


def isCaptured(piece, move, boardState):
    inactive = setActive(piece, boardState)[1]
    for opp in inactive:
        if boardState[opp][move[1]] == "1":
            return opp
    return False


def isThreatened(piece, boardState, dblMoveIndex):
    active, inactive = setActive(piece, boardState)
    threats = []
    for opp in inactive:
        captures = moves_postBlock(opp, boardState, False, dblMoveIndex)[1]
        caps = len(captures)
        if caps > 0:
            for i in range(caps):
                if active[piece][captures[i][1]] == "1":
                    threats.append(captures[i])
    return threats


def isCheck(active, boardState, dblMoveIndex):
    if "k" in active:
        if len(isThreatened("k", boardState, dblMoveIndex)) > 0:
            return "black in check", isThreatened("k", boardState, dblMoveIndex)
        else:
            return False
    elif "K" in active:
        if len(isThreatened("K", boardState, dblMoveIndex)) > 0:
            return "white in check", isThreatened("K", boardState, dblMoveIndex)
        else:
            return False


def moves_postBlock(piece, boardState, audit, dblMoveIndex):
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
    if audit:
        print(piece)
    edges = genEdges()
    all_positions = indexToPos(boardState[piece])
    legal, captures = [], []
    active, inactive = setActive(piece, boardState)
    for position in all_positions:
        if audit:
            print(position)
        if (piece == "P") or (piece == "p"):
            dbl = False
            startingPawns = pawn_starting_double(piece, boardState)
            fwd = []
            forwardSquare = MOVEMENTS[piece][0][0]
            captureSquares = MOVEMENTS[piece][1]
            if dblMoveIndex != None:
                if position not in edges:
                    if (bitIndex(position) + 1 == dblMoveIndex) or (
                        bitIndex(position) - 1 == dblMoveIndex
                    ):
                        if piece == "P":
                            captures.append((position, indexToTuple(dblMoveIndex + 8)))
                        if piece == "p":
                            captures.append((position, indexToTuple(dblMoveIndex - 8)))
                else:
                    if onBoard(indexToTuple(bitIndex(position) + 1)):
                        if bitIndex(position) + 1 == dblMoveIndex:
                            if piece == "P":
                                captures.append(
                                    (position, indexToTuple(dblMoveIndex + 8))
                                )
                            if piece == "p":
                                captures.append(
                                    (position, indexToTuple(dblMoveIndex - 8))
                                )
                    elif onBoard(indexToTuple(bitIndex(position) - 1)):
                        if bitIndex(position) - 1 == dblMoveIndex:
                            if piece == "P":
                                captures.append(
                                    (position, indexToTuple(dblMoveIndex + 8))
                                )
                            if piece == "p":
                                captures.append(
                                    (position, indexToTuple(dblMoveIndex - 8))
                                )
            fwd.append(get_newPos(position, forwardSquare))
            if bitIndex(position) in startingPawns:
                dbl = True
                if piece == "p":
                    fwd.append(get_newPos(position, (-2, 0)))
                elif piece == "P":
                    fwd.append(get_newPos(position, (2, 0)))
            cap1, cap2 = get_newPos(position, captureSquares[0]), get_newPos(
                position, captureSquares[1]
            )
            if dbl:
                if pieceInIndex(bitIndex(fwd[0]), boardState) == None:
                    if pieceInIndex(bitIndex(fwd[1]), boardState) == None:
                        legal.append((position, fwd[1]))
                    legal.append((position, fwd[0]))
            elif not dbl:
                if pieceInIndex(bitIndex(fwd[0]), boardState) == None:
                    legal.append((position, fwd[0]))
            if pieceInIndex(bitIndex(cap1), boardState) in inactive:
                captures.append((position, cap1))
            if pieceInIndex(bitIndex(cap2), boardState) in inactive:
                captures.append((position, cap2))
        else:
            for move_sect in MOVEMENTS[piece]:
                block, capture, block, edge, offBoard = (
                    False,
                    False,
                    False,
                    False,
                    False,
                )
                for i, move in enumerate(move_sect):
                    newPos = get_newPos(position, move)
                    if audit:
                        print(newPos)
                    if onBoard(newPos):
                        squareCheck = pieceInIndex(bitIndex(newPos), boardState)
                        if squareCheck != None:
                            if squareCheck in inactive:
                                if audit:
                                    print("capture")
                                [
                                    legal.append((position, get_newPos(position, mv)))
                                    for mv in move_sect[:i]
                                ]
                                captures.append((position, newPos))
                                capture = True
                                break
                            elif squareCheck in active:
                                if audit:
                                    print("block")
                                [
                                    legal.append((position, get_newPos(position, mv)))
                                    for mv in move_sect[:i]
                                ]
                                block = True
                                break
                        if newPos in edges:
                            if audit:
                                print("edge")
                            [
                                legal.append((position, get_newPos(position, mv)))
                                for mv in move_sect[: i + 1]
                            ]
                            edge = True
                            break
                    else:
                        if audit:
                            print("offboard")
                        offBoard = True
                        break
                if (not block) and (not capture) and (not edge) and (not offBoard):
                    if audit:
                        print("not")
                    [
                        legal.append((position, get_newPos(position, mv)))
                        for mv in move_sect
                    ]
    if audit:
        print("moves generated")
    legal_index, capture_index = [], []
    for pair in legal:
        if audit:
            print(pair)
        legal_index.append((bitIndex(pair[0]), bitIndex(pair[1])))
    for pair in captures:
        if audit:
            print(pair)
        capture_index.append((bitIndex(pair[0]), bitIndex(pair[1])))
    if audit:
        print(legal_index, capture_index)
    return legal_index, capture_index


def moves_postCheck(piece, boardState, audit, dblMoveIndex):
    legal1, capture1 = moves_postBlock(piece, boardState, audit, dblMoveIndex)
    legal2, capture2 = [], []
    active = setActive(piece, boardState)[0]
    for move in legal1:
        newboard = boardAfterMove(piece, move, boardState, audit)
        check = isCheck(active, newboard, dblMoveIndex)
        if not check:
            legal2.append(move)
    for move in capture1:
        newboard = boardAfterMove(piece, move, boardState, audit)
        check = isCheck(active, newboard, dblMoveIndex)
        if not check:
            capture2.append(move)
        # paste
    if audit:
        print("postCheck: ", legal2, capture2)
    return legal2, capture2


def boardAfterMove(piece, move, boardState, audit):
    # move is one of the tuples in lists provided by
    newBoard = copy.deepcopy(boardState)
    victim = isCaptured(piece, move, newBoard)
    if audit:
        print(boardState[piece], move[0])
    newBoard[piece] = changeBit(newBoard[piece], move[0], False)
    if audit:
        print(boardState[piece], move[0])
    newBoard[piece] = changeBit(newBoard[piece], move[1], True)
    if audit:
        print(boardState[piece], move[1])
    if victim != False:
        if audit:
            print(newBoard[victim], move[1])
        newBoard[victim] = changeBit(newBoard[victim], move[1], False)
    return newBoard


def pawn_starting_double(piece, boardState):
    starting_board = bit_initialise(STARTING_POSITIONS)[piece]
    current = boardState[piece]
    starting = []
    for i in range(64):
        if (current[i] == "1") & (starting_board[i] == "1"):
            starting.append(i)
    return starting


def pawnDoubleMove_index(color, board, move, dblMoves):
    if color == "w":
        if ("P", move) in dblMoves:
            return move[1]
    elif color == "b":
        if ("p", move) in dblMoves:
            return move[1]
    return None


def all_moves(activeColor, boardState, audit, dblMoveIndex):
    nonCaptures, captures = [], []
    for piece in setActiveByColor(activeColor, boardState):
        if audit:
            print(piece)
        moves = moves_postCheck(piece, boardState, False, dblMoveIndex)
        for a in moves[0]:
            if audit:
                print(a)
            nonCaptures.append((piece, a))
        for b in moves[1]:
            captures.append((piece, b))
    return nonCaptures, captures


def affectMove(color, move, boardState, audit, dblMoves, dblIndex):
    active = setActiveByColor(color, boardState)
    dblMoveIndex = pawnDoubleMove_index(color, boardState, move, dblMoves)
    moves, captures = (
        all_moves(color, boardState, audit, dblIndex)[0],
        all_moves(color, boardState, audit, dblIndex)[1],
    )
    for mv in moves:
        if mv[1] == move:
            board_after_move = boardAfterMove(mv[0], mv[1], boardState, audit)
    for mv in captures:
        if mv[1] == move:
            board_after_move = boardAfterMove(mv[0], mv[1], boardState, audit)
    return board_after_move, dblMoveIndex


def displayBoard(boardState, whiteOnTop):
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
    for piece in boardState:
        for pos in range(64):
            if boardState[piece][pos] == "1":
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
    if whiteOnTop:
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
    elif not whiteOnTop:
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


def file_rank_compile(color, filerank_input, boardState, dblMoveIndex, audit):
    for i in range(2):
        for move in all_moves(color, boardState, audit, dblMoveIndex)[i]:
            if filerank_index(filerank_input) == move[1]:
                return move
    return None


def available_moves_displayed(activeColor, boardState, audit, dblMoveIndex):
    allMoves = all_moves(activeColor, boardState, audit, dblMoveIndex)
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


def alternateColor(color):
    if color == COLORS[0]:
        return COLORS[1]
    elif color == COLORS[1]:
        return COLORS[0]


def fileRankInput(
    c,
    board,
    r,
    turnList,
    audit,
    dblMoveIndex,
    moveShow,
    displayShow,
    turnShow,
    whiteOnTop,
):
    if moveShow:
        print(
            "Available Moves:\n",
            available_moves_displayed(c, board, audit, dblMoveIndex),
        )
        newLine()
    if displayShow:
        displayBoard(board, whiteOnTop)
    if turnShow:
        (print(gameturn) for gameturn in turnList)
    inp = input(
        "M (Make Move) ||  P (Previous Moves) || A (Available Moves) || S (Stop Game)\n\t Action:\n\t "
    )
    newLine()
    if inp == "M":
        newLine()
        return input("Make Move: \n\t")
    elif inp == "P":
        (print(gameturn) for gameturn in turnList)
        newLine()
        return None
    elif inp == "A":
        print(
            "Available Moves:\n",
            available_moves_displayed(c, board, audit, dblMoveIndex),
        )
        newLine()
        return None
    elif inp == "S":
        r = False
        return None


def newLine():
    print("\n - - - - - - - - - - - - - - - - - - - - -\n")


def gameTurns(prevTurns, turns, color, text):
    sep = "||"
    if color == "w":
        prevTurns.append((turns.copy(), sep, text, "\n"))
    elif color == "b":
        prevTurns.append(("\t", text, "\n"))


def turnIncreaser(c, turns):
    if c == "w":
        return
    return turns + 1


def turnBlock(
    running,
    turnMoves,
    turnNumber,
    c,
    board,
    audit,
    dblMoves,
    init_dblIndex,
    whiteOnTop,
    moveShow,
    displayShow,
    turnShow,
):
    dblIndex = init_dblIndex
    input = fileRankInput(
        c,
        board,
        running,
        turnMoves,
        audit,
        init_dblIndex,
        moveShow,
        displayShow,
        turnShow,
        whiteOnTop,
    )
    if input == None:
        turnBlock(
            running,
            turnMoves,
            turnNumber,
            c,
            board,
            audit,
            dblMoves,
            init_dblIndex,
            whiteOnTop,
            moveShow,
            displayShow,
            turnShow,
        )
    move = file_rank_compile(c, input, board, init_dblIndex, audit)
    if move == None:
        print(ERROR_MESSAGE["move_unavailable"])
        turnBlock(
            running,
            turnMoves,
            turnNumber,
            c,
            board,
            audit,
            dblMoves,
            init_dblIndex,
            whiteOnTop,
            moveShow,
            displayShow,
            turnShow,
        )
    else:
        board, dblIndex = affectMove(c, move, board, audit, dblMoves, dblIndex)
        gameTurns(turnMoves, turnNumber, c, input)
        turnIncreaser(c, turnNumber)
        c = alternateColor(c)


# notation comprehension
def filerank_index(filerank):
    return bitIndex((int(filerank[1]) - 1, ord(filerank[0]) - ord("a")))


def index_filerank(index):
    tup = indexToTuple(index)
    return chr(tup[1] + ord("a")) + str(tup[0] + 1)


def gameConfig():
    inp = input("Show available Moves by default? Y/N\n")
    disp = input("Display Board after move? Y/N\n")
    prev = input("(Not recommended) Display Previous Moves By Default? Y/N\n")
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
