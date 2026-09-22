"""FastAPI app serving the board and one JSON move endpoint.

The server is authoritative about the rules. Every state payload carries the
legal-action list straight from ``State.legal_actions()``, so the browser
renders what it is told and never decides legality itself -- a second rules
implementation in JavaScript is exactly the thing that drifts out of agreement
with the Python and native engines.
"""
import json
import os
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .difficulty import Tiers
from .evaluator import OnnxEvaluator
from .sessions import SessionLimit, SessionStore

STATIC = Path(__file__).resolve().parent / "static"
MAX_BODY = 4096


class NewGame(BaseModel):
    difficulty: str | None = None
    human_side: int = Field(default=1, description="1 plays first (X), -1 plays second (O)")


class Move(BaseModel):
    action: int = Field(ge=0, le=80)


def serialize(state):
    return {
        "cells": list(state.cells),
        "boards": list(state.boards),
        "turn": state.turn,
        "forced": state.forced,
        "result": state.result,
        "legal": state.legal_actions(),
    }


def create_app(evaluator=None, tiers=None, store=None):
    # `is None`, not `or`: SessionStore defines __len__, so an empty store is
    # falsy and `store or SessionStore(...)` would quietly discard the one it
    # was handed and build a default-capped replacement.
    tiers = Tiers() if tiers is None else tiers
    evaluator = OnnxEvaluator.from_env() if evaluator is None else evaluator
    store = SessionStore(evaluator, tiers) if store is None else store

    app = FastAPI(title="Super Tic-Tac-Toe", docs_url=None, redoc_url=None)
    app.state.store = store
    app.state.tiers = tiers

    @app.middleware("http")
    async def cap_body(request: Request, call_next):
        declared = request.headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > MAX_BODY:
            raise HTTPException(status_code=413, detail="request body too large")
        return await call_next(request)

    def session_or_404(session_id: str):
        session = store.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="no such game; it may have expired")
        return session

    @app.get("/api/health")
    def health():
        model_info = {}
        model_path = getattr(evaluator, "path", "")
        sidecar = Path(f"{model_path}.json") if model_path else None
        if sidecar is not None and sidecar.exists():
            record = json.loads(sidecar.read_text())
            model_info = {"iteration": record.get("iteration"),
                          "arch": record.get("arch"),
                          "opset": record.get("opset")}
        return {
            "status": "ok",
            "model": model_info,
            "threads": int(os.environ.get("STTT_WEB_THREADS", 4)),
            "sessions": {"active": len(store), "max": store.max_sessions},
            "tiers": tiers.public(),
            "default_difficulty": tiers.default,
        }

    @app.post("/api/game", status_code=201)
    def new_game(body: NewGame):
        tier_key = body.difficulty or tiers.default
        if tier_key not in tiers:
            raise HTTPException(status_code=400, detail=f"unknown difficulty {tier_key!r}")
        if body.human_side not in (1, -1):
            raise HTTPException(status_code=400, detail="human_side must be 1 or -1")
        try:
            session_id, session = store.create(tier_key, body.human_side)
        except SessionLimit as limit:
            raise HTTPException(status_code=429, detail=str(limit)) from limit

        engine_action = None
        if session.human_side == -1:
            tier = tiers[tier_key]
            with session.lock:
                engine_action = session.play_engine(tier["simulations"], tier["leaf_batch"])
        return {"session": session_id, "state": serialize(session.state),
                "engine_action": engine_action, "backend": session.backend,
                "difficulty": tier_key}

    @app.get("/api/game/{session_id}")
    def read_game(session=Depends(session_or_404)):
        # Enough to rebuild the page after a refresh: whose seat, and every
        # move so far (X moves first, so sides alternate from index 0).
        return {"state": serialize(session.state), "difficulty": session.tier,
                "human_side": session.human_side, "history": list(session.history),
                "backend": session.backend}

    @app.post("/api/game/{session_id}/move")
    def play_move(body: Move, session=Depends(session_or_404)):
        tier = tiers[session.tier]
        with session.lock:
            if session.state.result is not None:
                raise HTTPException(status_code=409, detail="this game is already finished")
            try:
                session.play_human(body.action)
            except ValueError as error:
                # State.play validates; trusting the client's own filtering is
                # how an illegal move becomes a corrupted tree.
                raise HTTPException(status_code=400, detail=str(error)) from error
            engine_action = session.play_engine(tier["simulations"], tier["leaf_batch"])
        return {"state": serialize(session.state), "engine_action": engine_action,
                "difficulty": session.tier}

    @app.delete("/api/game/{session_id}", status_code=204)
    def end_game(session_id: str):
        store.drop(session_id)

    if STATIC.is_dir():
        app.mount("/assets", StaticFiles(directory=STATIC / "assets"), name="assets")

        @app.get("/")
        def index():
            return FileResponse(STATIC / "index.html")

    return app


app = None


def get_app():
    global app
    if app is None:
        app = create_app()
    return app


def main():
    import uvicorn
    uvicorn.run(get_app(), host="0.0.0.0", port=int(os.environ.get("STTT_WEB_PORT", 8000)))


if __name__ == "__main__":
    main()
