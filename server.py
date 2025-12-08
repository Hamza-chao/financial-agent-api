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
    version="3.0.0",
)

# Allow ALL origins, no credentials (simplest, safest for now)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # <-- wild-card origin
    allow_credentials=False,    # <-- important: must be False when using "*"
    allow_methods=["*"],
    allow_headers=["*"],
)
