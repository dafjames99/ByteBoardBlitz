from refactored_mechanics import Board

# r = True
# while r:
#     board = Board(config=[False, True, False])
#     while board.running:
#         board.preMove()
#         if (board.input != 'A') and (board.input != 'P') and (board.input != 'D'):
#             board.boardTurn()
#             board.afterTurn()
#         x = board.afterBoard()
# if not x:
#     r = False


board = Board(config=[False, True, False])
print(board.printAvailableMoves())
