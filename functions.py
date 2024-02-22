import ast

def extract_functions(file_path):
    functions = []

    with open(file_path, 'r') as file:
        tree = ast.parse(file.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)

    return functions

# Example usage
file_path = 'game_mechanisms.py'
function_list = extract_functions(file_path)
print(function_list)
