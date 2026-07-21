"""Training control and live telemetry.

``POST /train/start`` spawns the trainer as a subprocess; ``WS /ws/train`` streams
its metrics and logs. The gateway supervises rather than trains in-process: a
training run that segfaults or is killed must not take the inference service with
it, and the model the gateway serves is a *loaded checkpoint*, independent of
whatever is being trained.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect, status

from ..schemas import TrainStartRequest, TrainStatusResponse
from ..telemetry import RunAlreadyActive, TrainingSupervisor

logger = logging.getLogger("csllm.gateway.training")
router = APIRouter(tags=["training"])


def get_supervisor(request: Request) -> TrainingSupervisor:
    supervisor = getattr(request.app.state, "supervisor", None)
    if supervisor is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "training supervisor unavailable")
    return supervisor


@router.post("/train/start", response_model=TrainStatusResponse)
async def start_training(req: TrainStartRequest, request: Request) -> TrainStatusResponse:
    supervisor = get_supervisor(request)
    try:
        await supervisor.start(req.model_dump())
    except RunAlreadyActive as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    return TrainStatusResponse(**supervisor.status())


@router.post("/train/stop", response_model=TrainStatusResponse)
async def stop_training(request: Request) -> TrainStatusResponse:
    supervisor = get_supervisor(request)
    if not await supervisor.stop():
        raise HTTPException(status.HTTP_409_CONFLICT, "no training run is active")
    return TrainStatusResponse(**supervisor.status())


@router.post("/train/pause", response_model=TrainStatusResponse)
async def pause_training(request: Request) -> TrainStatusResponse:
    """SIGSTOP the run. It keeps its arena and optimizer state, so resume is exact."""
    supervisor = get_supervisor(request)
    if not await supervisor.pause():
        raise HTTPException(status.HTTP_409_CONFLICT, "no active run to pause")
    return TrainStatusResponse(**supervisor.status())


@router.post("/train/resume", response_model=TrainStatusResponse)
async def resume_training(request: Request) -> TrainStatusResponse:
    supervisor = get_supervisor(request)
    if not await supervisor.resume():
        raise HTTPException(status.HTTP_409_CONFLICT, "no paused run to resume")
    return TrainStatusResponse(**supervisor.status())


@router.get("/train/status", response_model=TrainStatusResponse)
async def training_status(request: Request) -> TrainStatusResponse:
    return TrainStatusResponse(**get_supervisor(request).status())


@router.websocket("/ws/train")
async def training_socket(websocket: WebSocket) -> None:
    supervisor = getattr(websocket.app.state, "supervisor", None)
    await websocket.accept()
    if supervisor is None:
        await websocket.send_json({"type": "error", "message": "supervisor unavailable"})
        await websocket.close()
        return

    # Status first, so a client that connects between runs still renders
    # something rather than waiting on an idle socket.
    await websocket.send_json({"type": "status", **supervisor.status()})
    try:
        async for event in supervisor.subscribe():
            await websocket.send_json(event)
    except WebSocketDisconnect:
        pass
    except Exception:  # pragma: no cover - defensive
        logger.exception("training websocket failed")
        with_close = getattr(websocket, "close", None)
        if with_close is not None:
            await websocket.close()
