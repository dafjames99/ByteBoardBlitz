from engine import GameBoard
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Keep one game in memory for now (later you can manage sessions/games by ID)
game = GameBoard()

class MoveRequest(BaseModel):
    san: str

@app.post("/new_game")
def new_game():
    global game
    game = GameBoard()
    return get_state()

class MakeMoveReq(BaseModel):
    san: Optional[str] = None
    from_sq: Optional[str] = Field(None, alias="from")
    to_sq: Optional[str] = Field(None, alias="to")
    class Config:
        allow_population_by_field_name = True
        
@app.post("/make_move")
async def make_move(move: MakeMoveReq, request: Request):
    """
    Accepts:
      - {"san": "e4"}
      - {"from": "e2", "to": "e4"}   (or "from_sq"/"to_sq")
    Logs the incoming raw JSON (for debugging) and uses parsed model values.
    """
    # --- Debug/logging: try to read raw body (safe because Starlette caches it) ---
    try:
        raw_body = await request.json()
    except Exception:
        raw_body = None
    print("make_move called. raw JSON:", raw_body)
    print("parsed model:", move.model_dump(by_alias=True))

    # --- Normalize fields (handle different client naming) ---
    san = move.san
    from_sq = move.from_sq
    to_sq = move.to_sq

    # Fallback to raw body values if the model didn't capture them (robustness)
    if not from_sq and isinstance(raw_body, dict):
        from_sq = raw_body.get("from") or raw_body.get("from_sq")
    if not to_sq and isinstance(raw_body, dict):
        to_sq = raw_body.get("to") or raw_body.get("to_sq")

    # --- Apply move ---
    try:
        if san:
            game.apply_san(san)
        elif from_sq and to_sq:
            # coords_to_san should accept "e2","e4" strings; adjust if it expects ints
            san_conv = game.coords_to_san(from_sq, to_sq)
            if not san_conv:
                raise HTTPException(status_code=400, detail="No legal move matches provided squares")
            game.apply_san(san_conv)
        else:
            raise HTTPException(status_code=400, detail="Missing move data: send 'san' or both 'from' and 'to'.")
    except HTTPException:  # re-raise fastapi HTTPExceptions
        raise
    except Exception as e:
        # return engine errors to client (useful for debugging)
        raise HTTPException(status_code=400, detail=str(e))

    return get_state()


@app.post("/undo")
def undo():
    global game
    try:
        game.undo()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return get_state()

@app.post("/redo")
def redo():
    global game
    try:
        game.redo()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return get_state()

@app.get("/state")
def get_state():
    fen = game.to_fen() if hasattr(game, "to_fen") else None
    return {
        "fen": fen,
        "turn": game.active_color,
        # "history": game.history, 
        'history': game.move_history,
        "canUndo": len(game.history) > 1,
        'canRedo': game.pointer != -1
    }
