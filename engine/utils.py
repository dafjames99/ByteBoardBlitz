from __future__ import annotations
import numpy as np

def fr_to_index(filerank: str | list[str]) -> int | list[int]:
    """Convert 'A1' → 0, 'H8' → 63."""
    # print(filerank)
    if isinstance(filerank, str):
        f, r = int(filerank[1]) - 1, ord(filerank[0].lower()) - 97
        return (int(filerank[1]) - 1) * 8 + (ord(filerank[0].lower()) - 97)
    elif isinstance(filerank, list):
        return [(int(fr[1]) - 1) * 8 + (ord(fr[0].lower()) - 97) for fr in filerank]

def index_to_fr(index: int | list[int]) -> str | list[str]:
    """Convert 0 → 'A1', 63 → 'H8'."""
    if isinstance(index, int):
        return f"{chr(97 + (index % 8)).upper()}{(index // 8) +1}"
    elif isinstance(index, list):
        return [f"{chr(97 + (i % 8)).upper()}{(i // 8) +1}" for i in index]
    
def formal_color(c):
    return 'White' if c == 'w' else 'Black'

def opp(c, formal = False):
    res = 'w' if c == 'b' else 'b'
    return formal_color(res) if formal else res

def ply_to_turn(ply: int):
    return (ply + 1) // 2