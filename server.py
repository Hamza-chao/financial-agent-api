# server.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Tuple, Optional
from dotenv import load_dotenv

from main import app as langgraph_app
from langserve import add_routes

# Load env (local .env for dev; on Render the env is already set)
load_dotenv()

app = FastAPI(
    title="AI Financial Analyst API",
    description="A conversational API for financial analysis and charting.",
    version="3.0.0",
)

# CORS: allow your frontend and local dev
origins = [
    "http://localhost:3000",
    "https://financial-agent-ui-theta.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LangServe routes (keep)
add_routes(
    app,
    langgraph_app,
    path="/agent",
    playground_type="default",
)

# ------------ Models ------------

class ChatRequest(BaseModel):
    question: str
    chat_history: List[Tuple[str, str]] = Field(default_factory=list)

class ChatResponse(BaseModel):
    text_response: str
    chart_image: Optional[str] = None

# ------------ In-memory state (demo only) ------------

GLOBAL_HISTORY: List[Tuple[str, str]] = []
GLOBAL_SYMBOLS: List[str] = []

# ------------ Routes ------------

@app.post("/chat", response_model=ChatResponse, summary="Chat with the Analyst")
async def chat_with_analyst(request: ChatRequest):
    """
    Main endpoint called by your Next.js UI.
    """
    global GLOBAL_SYMBOLS

    try:
        # 1) Save user message
        GLOBAL_HISTORY.append(("user", request.question))

        # 2) Call LangGraph app
        graph_input = {
            "question": request.question,
            "chat_history": GLOBAL_HISTORY,
            "stock_symbols": GLOBAL_SYMBOLS,
        }
        result = langgraph_app.invoke(graph_input)
        if not result:
            raise HTTPException(
                status_code=500,
                detail="Agent returned an empty result."
            )

        text = result.get("text_response", "No text response found.")
        chart = result.get("chart_image")

        # 3) Save assistant reply
        GLOBAL_HISTORY.append(("assistant", text))

        # 4) Update symbol memory
        new_symbols = result.get("stock_symbols")
        if new_symbols is not None:
            GLOBAL_SYMBOLS = new_symbols

        return ChatResponse(text_response=text, chart_image=chart)

    except Exception as e:
        print(f"An error occurred in /chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
async def reset_state():
    GLOBAL_HISTORY.clear()
    GLOBAL_SYMBOLS.clear()
    return {"status": "ok"}

