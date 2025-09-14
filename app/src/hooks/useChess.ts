// src/hooks/useChess.ts
"use client";

import { useState, useEffect } from "react";
import * as api from "../lib/api";

export function useChess() {
    const [game, setGame] = useState<any>(null);
    const [loading, setLoading] = useState(false);

    async function refresh() {
        const state = await api.getState();
        setGame(state);
        return state;
    }

    useEffect(() => {
        refresh().catch(console.error);
    }, []);

    async function move(from: string, to: string) {
        setLoading(true);
        try {
            const state = await api.makeMoveFromTo(from, to);
            setGame(state);
            return state;
        } finally {
            setLoading(false);
        }
    }

    async function newGame() {
        setLoading(true);
        try {
            const state = await api.newGame();
            setGame(state);
            return state;
        } finally {
            setLoading(false);
        }
    }

    async function undo() {
        setLoading(true);
        try {
            const state = await api.undo();
            setGame(state);
            return state;
        } finally {
            setLoading(false);
        }
    }

    async function redo() {
        setLoading(true);
        try {
            const state = await api.redo();
            setGame(state);
            return state;
        } finally {
            setLoading(false);
        }
    }

    return { game, move, newGame, undo, redo, refresh, loading };
}
