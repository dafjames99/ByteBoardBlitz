import mechanics as mc, scenarios as sc, machine as ai

SCENARIOS = sc.scenarios


def player_vs_player():
    """
    Captures the entire Game Loop
    """
    r = True
    config = mc.gameConfig()
    while r:
        board = mc.Board(config=config)
        while board.running:
            board.preMove()
            if (board.input != "A") and (board.input != "P") and (board.input != "D"):
                board.boardTurn()
                board.afterTurn()
        if board.running == False:
            board.afterBoard()
            if board.endAll:
                r = False
    board.__del__()


def static_scenario(scen):
    scenario = SCENARIOS[scen]
    board = mc.Board(config=[False, True, False])
    for move in scenario:
        board.manualMove(move)
        board.afterTurn()
    board.display()
    return board


# static_scenario("protections-captures-test")
# mc.getPieceEvaluation('', board.board, board.dblMoveIndex, board.active, board.inactive)


def machine_vs_machine(moveFunction, runForTurns):
    board = mc.Board(config=[False, True, False])
    i = board.turnNumber
    turns = []
    while i < (runForTurns * 2) + 1:
        captures, moves = board.all
        machineItem = moveFunction(captures + moves)
        board.manualMove(mc.getFilerankMove(machineItem[0], machineItem[1]))
        turns.append(mc.getFilerankMove(machineItem[0], machineItem[1]))
        board.afterTurn()
        i += 1
    board.display()
    print(turns)
    print(f"Checkmate? {board.checkmate}\nStalemate? {board.stalemate}")
    return board


# board = mc.Board(config=[False, True, False])
# captures, moves = board.all
# print(captures + moves)
