// src/components/Board.tsx
"use client";

import React, { useState, useMemo } from "react";
// import { Chessboard } from 'react-chessboard';

type BoardProps = {
    fen: string;
    onMove: (from: string, to: string) => Promise<void>;
};

const PIECE_UNICODE: Record<string, string> = {
    P: "♙", N: "♘", B: "♗", R: "♖", Q: "♕", K: "♔",
    p: "♟", n: "♞", b: "♝", r: "♜", q: "♛", k: "♚",
};

function parseFenBoard(fen: string): string[][] {
    const parts = fen.split(" ");
    const placement = parts[0] ?? "8/8/8/8/8/8/8/8";
    const ranks = placement.split("/");
    const board: string[][] = [];

    for (const rank of ranks) {
        const row: string[] = [];
        for (const ch of rank) {
            if (/\d/.test(ch)) {
                const n = parseInt(ch, 10);
                for (let i = 0; i < n; i++) row.push("");
            } else {
                row.push(ch);
            }
        }
        board.push(row);
    }
    return board;
}

function squareName(file: number, rankIndex: number) {
    // file: 0..7 -> a..h ; rankIndex: 0..7 -> rank 8..1
    const fileChar = String.fromCharCode("a".charCodeAt(0) + file);
    const rank = 8 - rankIndex;
    return `${fileChar}${rank}`;
}

export default function Board({ fen, onMove }: BoardProps) {
    const board = useMemo(() => parseFenBoard(fen), [fen]);
    const [selected, setSelected] = useState<string | null>(null);

    async function handleClick(file: number, rankIndex: number) {
        const sq = squareName(file, rankIndex);
        if (!selected) {
            setSelected(sq);
            return;
        }
        if (selected === sq) {
            setSelected(null);
            return;
        }
        // attempt move selected -> sq
        try {
            await onMove(selected, sq);
        } catch (e) {
            // let parent handle the error; keep selection for convenience
            console.error(e);
        } finally {
            setSelected(null);
        }
    }

    return (
        <div className="bg-white rounded shadow p-3">
            <div className="grid grid-cols-8 gap-0 w-[480px] h-[480px] md:w-[560px] md:h-[560px]">
                {board.map((row, rIdx) =>
                    row.map((cell, fIdx) => {
                        const isLight = (fIdx + rIdx) % 2 === 0;
                        const name = squareName(fIdx, rIdx);
                        const pieceChar = cell ? (PIECE_UNICODE[cell] ?? "") : "";
                        const isSelected = selected === name;
                        return (
                            <button
                                key={name}
                                onClick={() => handleClick(fIdx, rIdx)}
                                className={`w-full h-full flex items-center justify-center text-3xl md:text-4xl select-none
                  ${isLight ? "bg-yellow-50" : "bg-green-700/90 text-white"}
                  ${isSelected ? "ring-4 ring-indigo-400" : ""}
                `}
                                title={name}
                            >
                                <span>{pieceChar}</span>
                            </button>
                        );
                    })
                )}
            </div>
            <div className="text-xs text-neutral-500 mt-2">Click source square then target square to move.</div>
        </div>
    );
}
