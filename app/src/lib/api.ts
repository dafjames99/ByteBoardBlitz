// src/lib/api.ts
const BASE = "http://127.0.0.1:8000"; // adjust if your backend host/port differs

async function handleRes(res: Response) {
    if (!res.ok) {
        const text = await res.text();
        throw new Error(text || res.statusText);
    }
    return res.json();
}

export async function getState() {
    const res = await fetch(`${BASE}/state`);
    return handleRes(res);
}

export async function newGame() {
    const res = await fetch(`${BASE}/new_game`, { method: "POST" });
    return handleRes(res);
}

// We send from/to; backend should accept these fields and convert to your engine move.
export async function makeMoveFromTo(from: string, to: string) {
    const res = await fetch(`${BASE}/make_move`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ from, to }),
    });
    return handleRes(res);
}

export async function undo() {
    const res = await fetch(`${BASE}/undo`, { method: "POST" });
    return handleRes(res);
}

export async function redo() {
    const res = await fetch(`${BASE}/redo`, { method: "POST" });
    return handleRes(res);
}
