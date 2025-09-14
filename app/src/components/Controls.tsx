// src/components/Controls.tsx
"use client";

import React from "react";

type Props = {
    onNew: () => Promise<void>;
    onUndo: () => Promise<void>;
    onRedo: () => Promise<void>;
    canUndo: boolean;
    canRedo: boolean;
    loading?: boolean;
};

export default function Controls({ onNew, onUndo, onRedo, canUndo, canRedo, loading }: Props) {
    return (
        <div className="flex flex-col gap-3">
            <div className="flex gap-2">
                <button onClick={onNew} className="px-3 py-2 rounded bg-indigo-600 text-white">New game</button>
                <button onClick={onUndo} disabled={!canUndo} className="px-3 py-2 rounded bg-white border">
                    Undo
                </button>
                <button onClick={onRedo} disabled={!canRedo} className="px-3 py-2 rounded bg-white border">
                    Redo
                </button>
            </div>
            <div className="text-sm text-gray-600">{loading ? "Processing..." : ""}</div>
        </div>
    );
}
