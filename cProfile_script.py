import cProfile
import pstats


if __name__ == "__main__":
    # Run your main function with cProfile
    cProfile.run('gameLoop()', 'profile_stats')

    # Create a Stats object from the profiling results
    stats = pstats.Stats('profile_stats')

    # Optionally, you can sort the results by various criteria
    stats.sort_stats('cumulative')  # Sort by cumulative time

    # Print the profiling results

    stats.print_stats()