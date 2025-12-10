# main.py
import os
import io
import base64
from typing import List, Optional
from typing_extensions import TypedDict
import re
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yfinance as yf
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*Ticker.earnings.*",
    category=DeprecationWarning,
)

load_dotenv("/etc/secrets/.env")


# ==============================================================================
# 1. State and Tool Definitions
# ==============================================================================

class GraphState(TypedDict):
    """Represents the state of our graph."""
    chat_history: List[tuple[str, str]]
    question: str
    stock_symbols: List[str]
    chart_image: Optional[str]
    text_response: str

class StockSymbols(BaseModel):
    """
    Extracts a list of stock ticker symbols from text.
    IMPORTANT: If a company name is mentioned (e.g., 'Microsoft', 'Apple'), extract its corresponding ticker symbol (e.g., 'MSFT', 'AAPL').
    """
    symbols: Optional[List[str]] = Field(None, description="A list of stock ticker symbols.")

# ==============================================================================
# 2. Tool Functions
# ==============================================================================
COMMON_NAME_TO_TICKER = {
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "apple": "AAPL",
    "microsoft": "MSFT",
    "meta": "META",
    "facebook": "META",
    "tesla": "TSLA",
    "coca cola": "KO",
    "coca-cola": "KO",
    "nvidia": "NVDA",
}


def fetch_stock_data(stock_symbol: str):
    print(f"---yfinance: fetching price history for {stock_symbol}---")
    try:
        ticker = yf.Ticker(stock_symbol)
        hist = ticker.history(period="6mo", interval="1d")
        print(f"history rows for {stock_symbol}: {len(hist)}")
        if hist.empty:
            print(f"yfinance: no data for {stock_symbol}")
            return None
        hist = hist.dropna(subset=["Close"])
        return hist["Close"]
    except Exception as e:
        print(f"yfinance error for {stock_symbol}: {e}")
        return None


def generate_chart_tool(stock_symbols: List[str]):
    """Generates a base64-encoded PNG image of stock performance."""
    print(f"---Tool: Generating Chart for {stock_symbols}---")

    plt.style.use('dark_background')
    fig, ax = plt.subplots()
    has_data = False

    for symbol in stock_symbols:
        price_history = fetch_stock_data(symbol)
        if price_history is None or getattr(price_history, "empty", False):
            print(f"Could not retrieve price data for {symbol}, skipping.")
            continue

        has_data = True
        # Normalize to % change from first point (same as before)
        normalized_prices = (price_history / price_history.iloc[0] - 1) * 100
        normalized_prices.plot(ax=ax, label=symbol)

    if has_data:
        if len(stock_symbols) == 1:
            ax.set_title(f"{stock_symbols[0]} Stock Performance")
        else:
            ax.set_title("Stock Performance Comparison")
        ax.set_xlabel("Date")
        ax.set_ylabel("Percentage Change (%)")
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    else:
        ax.set_title("No Data Available")
        ax.text(
            0.5, 0.5,
            "Could not retrieve stock data.",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)
    return image_base64

def get_latest_earnings_tool(stock_symbol: str):
    """Fetch the latest quarterly EPS in a structured way."""
    print(f"---Tool: Fetching Earnings for {stock_symbol} (yfinance)---")

    try:
        ticker = yf.Ticker(stock_symbol)
        earnings = ticker.quarterly_earnings  # DataFrame

        if earnings is None or earnings.empty:
            return {
                "available": False,
                "symbol": stock_symbol,
                "fiscal_end": None,
                "eps": None,
            }

        latest = earnings.iloc[-1]
        fiscal_end = latest.name
        reported_eps = latest.get("Earnings")

        return {
            "available": True,
            "symbol": stock_symbol,
            "fiscal_end": str(fiscal_end),
            "eps": float(reported_eps) if reported_eps is not None else None,
        }
    except Exception as e:
        print(f"earnings error for {stock_symbol}: {e}")
        return {
            "available": False,
            "symbol": stock_symbol,
            "fiscal_end": None,
            "eps": None,
        }


def get_company_overview_tool(stock_symbol: str):
    """Fetches company overview data, including key metrics and description using yfinance."""
    print(f"---Tool: Fetching Company Overview for {stock_symbol} (yfinance)---")

    try:
        ticker = yf.Ticker(stock_symbol)
        info = ticker.info  # dict with fundamentals

        if not info:
            return f"ERROR: No overview data found for {stock_symbol}."

        key_metrics = {
            "Description": info.get("longBusinessSummary"),
            "MarketCapitalization": info.get("marketCap"),
            "PERatio": info.get("trailingPE"),
            "EPS": info.get("trailingEps"),
            "52WeekHigh": info.get("fiftyTwoWeekHigh"),
            "52WeekLow": info.get("fiftyTwoWeekLow"),
        }
        return f"Company Overview for {stock_symbol}:\n{key_metrics}"
    except Exception as e:
        return f"An unexpected error occurred while fetching overview for {stock_symbol}: {e}"
# ==============================================================================
# 3. Graph Nodes
# ==============================================================================
def extract_symbols_from_text(text: str) -> List[str]:
    """Extract ticker symbols from arbitrary text using LLM + regex + name heuristics."""
    text = text.strip()
    if not text:
        return []

    prompt = f"""From the following text, extract ALL stock ticker symbols you can find.

If a company name is used instead of a ticker, convert it to the most common ticker symbol, e.g.:
- Apple -> AAPL
- Microsoft -> MSFT
- Google or Alphabet -> GOOGL
- Meta or Facebook -> META
- AT&T or ATT -> T

Return ONLY the ticker symbols, as a JSON-compatible list like ["AAPL", "MSFT"].

Text:
---
{text}
---
"""

    llm_with_tool = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools([StockSymbols])
    response = llm_with_tool.invoke(prompt)

    llm_symbols: list[str] = []
    if getattr(response, "tool_calls", None):
        llm_symbols = response.tool_calls[0]["args"].get("symbols", []) or []

    # Regex fallback
    regex_pattern = r"\b[A-Z]{2,5}\b"
    raw_regex_symbols = re.findall(regex_pattern, text)
    blacklist = {"EPS", "API", "PE", "PES", "ETF", "IPO", "HTTP", "JSON", "AT", "AI"}
    regex_symbols = [s for s in raw_regex_symbols if s not in blacklist]

    # Heuristic: map common names → tickers
    name_symbols: list[str] = []
    lower = text.lower()
    for name, ticker in COMMON_NAME_TO_TICKER.items():
        if name in lower:
            name_symbols.append(ticker)

    combined = llm_symbols + regex_symbols + name_symbols
    final = sorted(list(set(combined)))

    print(f"---extract_symbols_from_text | LLM: {llm_symbols} | Regex: {regex_symbols} | Names: {name_symbols} | Final: {final}---")
    return final


def single_stock_agent_node(state: GraphState):
    """Provides a detailed, synthesized analysis of a single stock, including a table."""
    print("---Node: Single Stock Agent---")
    question = state["question"]
    symbol = state["stock_symbols"][0]

    # --- Raw tools ---
    earnings_struct = get_latest_earnings_tool(symbol)
    overview_context = get_company_overview_tool(symbol)
    chart_image = generate_chart_tool(state["stock_symbols"])

    # Build a deterministic earnings summary in code
    if earnings_struct["available"] and earnings_struct["eps"] is not None:
        earnings_available = True
        earnings_summary = (
            f"Latest reported quarterly EPS for {symbol}: "
            f"{earnings_struct['eps']} (fiscal period ending {earnings_struct['fiscal_end']})."
        )
    else:
        earnings_available = False
        earnings_summary = (
            f"Recent quarterly EPS data for {symbol} is not available from the current data source. "
            "Investors typically focus on this company's revenue growth, margins, and guidance "
            "when evaluating its earnings."
        )

    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

    prompt = f"""You are a sophisticated financial analyst AI. Your goal is to provide a synthesized analysis for the user.
A performance chart is being generated and displayed to the user alongside your text response.
Use the following data to create your analysis. Format your response using clear markdown headings.

## Summary
Start with a brief summary that directly answers the user's question about recent performance.
Do not talk about missing data, APIs, or limitations.

## Key Metrics
Create a markdown table summarizing the key metrics from the company overview. Include: Market Cap, P/E Ratio, EPS, 52-Week High, and 52-Week Low.

## Earnings Performance
Start this section by **copying the following sentence EXACTLY** as the first paragraph:

{earnings_summary}

After that first paragraph, you may optionally add 1–2 sentences giving additional investment context
(e.g., how investors might interpret these earnings relative to the business or macro conditions).

Do **NOT** say that earnings data are missing, not released yet, or unavailable if the sentence above
already states that EPS is known. Do **NOT** contradict the given EPS value or fiscal date.

## Overall Analysis & Chart Context
In your concluding thoughts, briefly describe the company's business. Then, explicitly address the
relationship between the financial data and the stock's price trend visible in the chart. Assume the chart
shows the last 3–6 months of performance.

Do not mention tools, APIs, rate limits, or data sources in your answer.

---
EARNINGS_AVAILABLE_FLAG: {earnings_available}

RAW COMPANY OVERVIEW DATA:
{overview_context}

---
USER'S QUESTION: "{question}"
"""

    analysis_text = llm.invoke(prompt).content
    return {"text_response": analysis_text, "chart_image": chart_image}


def extract_symbols_node(state: GraphState):
    """
    Determines the FINAL list of symbols to analyze now, based on the previous
    symbols in state + the latest question.

    Behavior:
    - "add/include/also/plus" + new tickers                  -> ADD (previous ∪ new)
    - "compare/versus/vs/against" + pronoun ("it"/"them")    -> ADD (if previous_symbols exist)
    - new tickers otherwise                                  -> REPLACE (new only)
    - no new tickers                                         -> KEEP_PREVIOUS
    """
    print("---Node: Extracting Symbols---")

    question = state["question"]
    previous_symbols = state.get("stock_symbols") or []

    # Symbols in the latest user question only
    new_symbols = extract_symbols_from_text(question)
    q_lower = question.lower()

    # Patterns
    add_words_pattern = r"\b(add|include|also|plus)\b"
    compare_words_pattern = r"\b(compare|versus|vs|against)\b"
    pronoun_pattern = r"\b(it|them|this|that)\b"

    # 1) Explicit add/include/also/plus -> always ADD
    if new_symbols and re.search(add_words_pattern, q_lower):
        final_symbols = sorted(list(set(previous_symbols + new_symbols)))
        mode = "ADD (explicit add/include/also/plus)"

    # 2) "compare it/them to ..." with previous symbols -> ADD
    elif (
        new_symbols
        and previous_symbols
        and re.search(compare_words_pattern, q_lower)
        and re.search(pronoun_pattern, q_lower)
    ):
        # e.g. first: "What are earnings for NVDA?"
        #      then:  "compare it to google"  -> NVDA + GOOGL
        final_symbols = sorted(list(set(previous_symbols + new_symbols)))
        mode = "ADD (compare + pronoun, using previous)"

    # 3) We have new tickers but no explicit add semantics -> REPLACE
    elif new_symbols:
        # e.g. "Compare AAPL and MSFT" (even if previous_symbols existed)
        final_symbols = new_symbols
        mode = "REPLACE"

    # 4) No new tickers -> keep previous
    else:
        final_symbols = previous_symbols
        mode = "KEEP_PREVIOUS"

    print(f"---Mode: {mode} | Previous: {previous_symbols} | New: {new_symbols} | Final: {final_symbols}---")
    return {"stock_symbols": final_symbols}






def comparison_node(state: GraphState):
    """Compares multiple stocks based on their latest fundamentals and presents a table."""
    print("---Node: Comparison Agent---")
    question = state['question']
    stock_symbols = state['stock_symbols']

    all_context = []
    for symbol in stock_symbols:
        # Remove this:
        # earnings = get_latest_earnings_tool(symbol)
        overview = get_company_overview_tool(symbol)
        all_context.append(
            f"--- Data for {symbol} ---\nOverview: {overview}"
        )

    combined_context = "\n\n".join(all_context)
    comparison_chart = generate_chart_tool(stock_symbols)

    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

    prompt = f"""You are a helpful financial analyst. The user wants to compare {', '.join(stock_symbols)}.

Use the following data to generate a response. Format your response using clear markdown headings.

## Comparison Table
Create a markdown table with the following columns: Company (Symbol), Market Cap, P/E Ratio, and EPS.
Use EPS from the company overview data.

## Summary
After the table, write a brief, objective summary paragraph that compares the companies based on the data.

If any data contains an API error (e.g., it literally includes the words 'rate limit' or 'ERROR'), do the following:
- Exclude those tickers from the comparison table.
- After the summary, add ONE short sentence like:
  "Data for the following tickers could not be retrieved right now: TICKER1, TICKER2. Please try again later."
If there are NO such errors, DO NOT talk about APIs, errors, rate limits, or say that everything worked fine. Just give the table and the summary.

---
COMBINED DATA:
{combined_context}
---
USER'S ORIGINAL QUESTION: "{question}"
"""

    analysis_text = llm.invoke(prompt).content
    return {"text_response": analysis_text, "chart_image": comparison_chart}


def general_finance_node(state: GraphState):
    """Handles general financial questions without specific stock symbols."""
    print("---Node: General Finance---")
    question = state['question']
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    
    prompt = f"""You are an AI Financial Analyst. The user has asked a general investment question: "{question}"

    Your task is to provide a helpful, educational response without giving direct financial advice. 
    Suggest a few well-known, large-cap companies from different sectors as DIVERSIFIED EXAMPLES. For each company, briefly state its sector and what it's known for.

    For example:
    - A technology leader like Alphabet (GOOGL), which dominates online search and cloud computing.
    - A consumer staples giant like Procter & Gamble (PG), known for its portfolio of household brands.
    - A major financial institution like JPMorgan Chase (JPM), a leader in banking and financial services.

    IMPORTANT: Conclude your response with a clear disclaimer that you are an AI, this is not financial advice, and the user should consult with a qualified financial advisor before making any investment decisions.
    """
    
    response = llm.invoke(prompt).content
    return {"text_response": response}

def clarification_node(state: GraphState):
    """Asks the user for clarification when no symbols are found."""
    return {"text_response": "I couldn't identify any stock symbols in your message. Which stock(s) are you interested in?"}

# ==============================================================================
# 4. Graph Router
# ==============================================================================

def route_after_extraction(state: GraphState):
    """Routes to the correct node based on the number of symbols and user intent."""
    question = state['question']
    num_symbols = len(state.get("stock_symbols", []))
    
    if num_symbols == 0:
        classification_llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        classification_prompt = f"""Classify the user's question into one of the following categories:
        1. 'general_investment_question': User is asking for investment ideas, suggestions, or general financial advice.
        2. 'clarification_needed': User's query is unclear, a greeting, or not a financial question.

        User's question: "{question}"
        
        Category:"""
        
        response = classification_llm.invoke(classification_prompt).content.strip()
        
        if "general_investment_question" in response:
            print("---Routing: General finance question---")
            return "general_finance_agent"
        else:
            print("---Routing: Clarification needed---")
            return "clarification"
            
    print(f"---Routing: Found {num_symbols} symbols---")
    if num_symbols == 1:
        return "single_stock_agent"
    else:
        return "comparison_agent"

# ==============================================================================
# 5. Graph Definition
# ==============================================================================

workflow = StateGraph(GraphState)

# Add all the nodes
workflow.add_node("extract_symbols", extract_symbols_node)
workflow.add_node("single_stock_agent", single_stock_agent_node)
workflow.add_node("comparison_agent", comparison_node)
workflow.add_node("general_finance_agent", general_finance_node)
workflow.add_node("clarification", clarification_node)

# Define the graph's flow
workflow.set_entry_point("extract_symbols")

workflow.add_conditional_edges(
    "extract_symbols",
    route_after_extraction,
    {
        "single_stock_agent": "single_stock_agent",
        "comparison_agent": "comparison_agent",
        "general_finance_agent": "general_finance_agent",
        "clarification": "clarification"
    }
)

workflow.add_edge("single_stock_agent", END)
workflow.add_edge("comparison_agent", END)
workflow.add_edge("general_finance_agent", END)
workflow.add_edge("clarification", END)

# Compile the graph into a runnable app
app = workflow.compile()
