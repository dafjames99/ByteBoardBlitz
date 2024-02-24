import cProfile
import pstats
from refactored_mechanics import Board

def mainLoop():
    r = True
    while r:
    board = Board(config=[False, True, False])
        while board.running:
            board.preMove() 
            board.boardTurn()

    board.afterTurn()
    # x = board.afterBoard()
    # if not x:
    #     r = False

if __name__ == "__main__":
    # Run your main function with cProfile
    cProfile.run('mainLoop()', 'profile_stats')

    # Create a Stats object from the profiling results
    stats = pstats.Stats('profile_stats')

    # Optionally, you can sort the results by various criteria
    stats.sort_stats('cumulative')  # Sort by cumulative time

    # Print the profiling results

    stats.print_stats()

