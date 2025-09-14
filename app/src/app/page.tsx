"use client"; // if using Next.js App Router

import { useState, useEffect } from "react";
import { Chessboard } from "react-chessboard";

export default function Home() {
  const [fen, setFen] = useState("start"); // default start position
  const [gameId, setGameId] = useState(null); // optional if you manage multiple games

  // Fetch the initial game state from backend
  useEffect(() => {
    fetch("http://127.0.0.1:8000/state")
      .then((res) => res.json())
      .then((data) => setFen(data.fen));
  }, []);

  // Called whenever a piece is dropped
  const onDrop = async (sourceSquare: string, targetSquare: string) => {
    console.log("Move attempted:", sourceSquare, targetSquare);

    try {
      const res = await fetch("http://127.0.0.1:8000/make_move", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ from: sourceSquare, to: targetSquare }),
      });

      if (!res.ok) {
        const err = await res.json();
        alert("Illegal move: " + err.detail);
        return false;
      }
      await console.log(res.json());
      const data = await res.json();
      setFen(data.fen); // backend returns updated FEN
      return true;
    } catch (err) {
      console.error("Move error:", err);
      return false;
    }
  };

  return (
    <div className="flex justify-center items-center h-screen/1.5 bg-gray-100">
      <Chessboard
        id="Chessboard"
        position={fen}
        onPieceDrop={onDrop}
        boardWidth={500}
      />
    </div>
  );
}
