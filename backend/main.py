# main.py - the web server. Angular talks to it over a WebSocket.
# Run (from the project folder):   uvicorn backend.main:app --port 8000
#
# WHY A WEBSOCKET (instead of a normal HTTP POST)?
# - One connection stays open, and the server can PUSH many messages: progress steps, sources, tokens, done.
# - Two-way: the browser can send the next question (or "reset") on the same connection.
# - The connection itself is the "chat session", so the server keeps the history per connection (see `history` below).
#
# Message protocol (JSON):
#   Browser -> server:  {"type": "ask", "question": "...", "mode": "pipeline" | "agent"}   or   {"type": "reset"}
#   Server -> browser:  {"type": "step", ...} {"type": "sources", ...} {"type": "token", ...} {"type": "blocked", ...}
#                       {"type": "error", ...} {"type": "done"} {"type": "reset_done"}

import logging  # Python's built-in logger (like console.error, but with levels and timestamps).

from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # FastAPI = Python's NestJS-like web framework.
from pydantic import ValidationError  # Raised when a browser message has the wrong shape.
from starlette.concurrency import iterate_in_threadpool  # Runs our normal (blocking) generator in a worker thread.

from backend import config  # MAX_HISTORY_MESSAGES.
from backend.chat.agent import run_agent  # Agent mode.
from backend.chat.pipeline import run_pipeline  # Pipeline mode (the diagram).
from backend.models import ClientMessage  # Shape of browser messages.

logging.basicConfig(level=logging.INFO)  # Show INFO logs and above in the terminal / Azure log stream.
logger = logging.getLogger("chatbot")  # A named logger for this app.

app = FastAPI(title="RAG chatbot on Azure")  # Create the app (like NestFactory.create).


@app.get("/api/health")  # GET /api/health - used by Azure App Service health checks.
def health() -> dict:  # Returning a dict = FastAPI sends JSON.
    return {"status": "ok", "provider": config.AI_PROVIDER, "chat_model": config.CHAT_MODEL_NAME, "embedding_model": config.EMBEDDING_MODEL_NAME}


@app.websocket("/ws/chat")  # The WebSocket endpoint: ws://localhost:8000/ws/chat
async def chat_socket(websocket: WebSocket) -> None:  # `async def` = an async function, like `async` in TS.
    await websocket.accept()  # Accept the connection (the WebSocket "handshake").
    history: list[dict] = []  # THIS connection's conversation. Each browser tab gets its own list.
    try:
        while True:  # Keep serving questions until the browser disconnects.
            raw = await websocket.receive_json()  # Wait for the next message from Angular.
            try:
                message = ClientMessage.model_validate(raw)  # Validate it (like a DTO with class-validator).
            except ValidationError:  # Wrong shape?
                await websocket.send_json({"type": "error", "message": "Invalid message format."})  # Tell the browser.
                continue  # Wait for the next message.

            if message.type == "reset":  # "New chat" button.
                history.clear()  # Forget the conversation (the LLM itself never remembered anything).
                await websocket.send_json({"type": "reset_done"})  # Confirm.
                continue  # Wait for the next message.

            runner = run_agent if message.mode == "agent" else run_pipeline  # Pick the strategy. Functions are values in Python too.
            recent_history = history[-config.MAX_HISTORY_MESSAGES:]  # Only the last N messages (a copy, so the runner can't change ours).
            answer_parts: list[str] = []  # Collect the tokens to store the full answer in history afterwards.
            try:
                events = runner(message.question, recent_history)  # Create the generator (nothing runs yet - it's lazy).
                async for event in iterate_in_threadpool(events):  # Run it in a thread so other users' sockets are not blocked.
                    if event["type"] == "token":  # A piece of the answer?
                        answer_parts.append(event["text"])  # Remember it.
                    await websocket.send_json(event)  # Push the event to Angular immediately.
            except WebSocketDisconnect:  # Browser closed while we were answering.
                raise  # Let the outer handler deal with it.
            except Exception:  # Anything else (Azure timeout, quota/429 errors, bug...).
                logger.exception("Answering failed")  # Full stack trace in the logs (Application Insights in Azure).
                await websocket.send_json({"type": "error", "message": "Something went wrong. Please try again."})  # Never send internals to users.
                await websocket.send_json({"type": "done"})  # Unblock the UI.
                continue  # Keep the connection alive for the next question.

            if answer_parts:  # Only store real answers (not blocked questions).
                history.append({"role": "user", "content": message.question})  # The question...
                history.append({"role": "assistant", "content": "".join(answer_parts)})  # ...and the full answer.
    except WebSocketDisconnect:  # The user closed the tab or lost connection.
        logger.info("Client disconnected")  # Normal - nothing to clean up, `history` simply disappears.
