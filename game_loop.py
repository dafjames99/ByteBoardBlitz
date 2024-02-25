from mechanics import Board
import mechanics as mc
import scenarios as sc
import machine as ai

SCENARIOS = sc.scenarios

def player_vs_player():
    """
    Captures the entire Game Loop
    """
    r = True
    config = mc.gameConfig()
    while r:
        board = Board(config=config)
        while board.running:
            board.preMove()
            if (board.input != 'A') and (board.input != 'P') and (board.input != 'D'):
                board.boardTurn()
                board.afterTurn()
        if board.running == False:
            board.afterBoard()
            if board.endAll:
                r = False
    board.__del__()

def static_scenario(scen):
    scenario = SCENARIOS[scen]
    board = Board(config=[False, True, False])
    for move in scenario:
        board.manualMove(move)
        board.afterTurn()
    board.display()
    return board

