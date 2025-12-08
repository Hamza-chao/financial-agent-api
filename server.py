# server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from main import app as langgraph_app
from langserve import add_routes
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Tuple, Optional
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="AI Financial Analyst API",
    description="A conversational API for financial analysis and charting.",
    version="3.0.0"
)

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="AI Financial Analyst API",
    description="A conversational API for financial analysis and charting.",
    version="3.0.0",
)

# Temporarily allow all origins (you can restrict later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # 👈 key change
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

class ChatRequest(BaseModel):
    question: str
    # you can ignore this in the backend now
    chat_history: List[Tuple[str, str]] = Field(default_factory=list)

class ChatResponse(BaseModel):
    text_response: str
    chart_image: Optional[str] = None


GLOBAL_HISTORY: List[Tuple[str, str]] = []
GLOBAL_SYMBOLS: List[str] = [] 


@app.post("/chat", response_model=ChatResponse, summary="Chat with the Analyst")
async def chat_with_analyst(request: ChatRequest):
    global GLOBAL_SYMBOLS

    try:
        # 1) Store the new user message in history
        GLOBAL_HISTORY.append(("user", request.question))

        # 2) Call the LangGraph app with full history AND previous stock_symbols
        graph_input = {
            "question": request.question,
            "chat_history": GLOBAL_HISTORY,
            "stock_symbols": GLOBAL_SYMBOLS,  # <--- pass previous set in
        }
        result = langgraph_app.invoke(graph_input)
        if not result:
            raise HTTPException(status_code=500, detail="Agent returned an empty result.")

        text = result.get("text_response", "No text response found.")
        chart = result.get("chart_image")

        # 3) Store the assistant reply
        GLOBAL_HISTORY.append(("assistant", text))

        # 4) Update the GLOBAL_SYMBOLS with what the graph decided this turn
        new_symbols = result.get("stock_symbols")
        if new_symbols is not None:
            GLOBAL_SYMBOLS = new_symbols

        return ChatResponse(text_response=text, chart_image=chart)

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/reset")
async def reset_state():
    GLOBAL_HISTORY.clear()
    GLOBAL_SYMBOLS.clear()
    return {"status": "ok"}
