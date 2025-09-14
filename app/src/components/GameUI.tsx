// src/components/GameUI.tsx
"use client";
import React from "react";
import Board from "./Board";
import Controls from "./Controls";
import { useChess } from "../hooks/useChess";


export default function GameUI() {
    const { game, move, newGame, undo, redo, loading } = useChess();

    return (
        <div className="w-full flex flex-col md:flex-row gap-6">
            <div>
                <Board fen={game?.fen ?? "8/8/8/8/8/8/8/8 w - - 0 1"} onMove={async (from, to) => {
                    try {
                        await move(from, to);
                    } catch (e: any) {
                        alert(e?.message ?? "Move failed");
                    }
                }} />
            </div>

            <aside className="flex-1">
                <div className="mb-4">
                    <Controls
                        onNew={newGame}
                        onUndo={undo}
                        onRedo={redo}
                        canUndo={game?.canUndo ?? false}
                        canRedo={game?.canRedo ?? false}
                        loading={loading}
                    />
                </div>

                <section className="bg-white p-4 rounded shadow">
                    <h3 className="font-semibold mb-2">Game info</h3>
                    <div className="text-sm">
                        <div><strong>FEN:</strong> <code className="break-all">{game?.fen ?? "—"}</code></div>
                        <div><strong>Turn:</strong> {game?.turn ?? "—"}</div>
                        <div className="mt-2"><strong>History:</strong></div>
                        <ol className="list-decimal ml-6 mt-1 text-sm max-h-40 overflow-auto">
                            {game?.history?.map((m: string, i: number) => <li key={i}>{m}</li>)}
                        </ol>
                    </div>
                </section>
            </aside>
        </div>
    );
}
