import scenarios as sc, pstats, cProfile, ast

SCENARIOS = sc.scenarios

def profile_code(function_string):
    if __name__ == "__main__":
        cProfile.run(f'{function_string}', 'profile_stats')
        stats = pstats.Stats('profile_stats')
        stats.sort_stats('cumulative')  # Sort by cumulative time
        stats.print_stats()
def extract_functions(code_filepath):
    functions = []
    with open(code_filepath, 'r') as file:
        tree = ast.parse(file.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
    return functions

