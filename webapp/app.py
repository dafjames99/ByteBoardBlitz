from flask import Flask, render_template
import sys
import os

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
venv_dir = os.path.join(parent_dir, "venv")
sys.path.append(parent_dir)
sys.path.append(venv_dir)

from mechanics import Board, bit_to_indices, index_filerank
from game_loop import static_scenario
from initial_vars import (
    outputFormat as OUTPUT_FORMAT,
    indexToFilerank as INDEX_FILERANK,
)

app = Flask(__name__)
board = static_scenario("4-move-checkmate")


def prepareAppOutput(board):
    output = {}
    for piece in board:
        loc = bit_to_indices(piece, board)
        for l in loc:
            output[INDEX_FILERANK[l]] = OUTPUT_FORMAT[piece]
    return output


@app.route("/")
def main():
    board = Board(config=[False, True, False])
    boardstate = prepareAppOutput(board.board)
    return render_template("index.html", boardstate=boardstate)


if __name__ == "__main__":
    app.run(debug=True)
