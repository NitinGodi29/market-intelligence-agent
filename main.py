from fastapi import FastAPI, Response
from ingest import (
    conditional_edge,
    get_ticker_details,
    return_quite_market_response,
    generate_summary_using_llm,
    save_summary_to_db,
)
from pydantic import BaseModel
from typing import List, Dict, Optional
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict


class TickerInformation(BaseModel):
    ticker: str
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    change_percent: float
    news_headlines: list[str]
    is_volatile: bool
    daily_sentiment_score: float


class ResponseData(BaseModel):
    ticker: str
    details: Optional[List[TickerInformation]]
    executive_summary: Optional[str]
    recommendation: Optional[str]
    confidence_score: Optional[float]
    data_status: Optional[str]


class GraphState(TypedDict):
    ticker: str
    ticker_details: Optional[Dict]
    executive_summary: Optional[str]
    recommendation: Optional[str]
    confidence_score: Optional[float]
    data_status: Optional[str]


app = FastAPI()


@app.get("/analyze/{ticker}")
async def analyze_ticker(ticker: str, response: Response):
    try:
        builder = StateGraph(GraphState)
        builder.add_node("fetch_details", get_ticker_details)
        builder.add_node("generate_summary", generate_summary_using_llm)
        builder.add_node("quite_market", return_quite_market_response)
        builder.add_node("save_summary", save_summary_to_db)

        builder.add_edge(START, "fetch_details")
        builder.add_conditional_edges("fetch_details", conditional_edge)
        builder.add_edge("generate_summary", "save_summary")
        builder.add_edge("quite_market", "save_summary")
        builder.add_edge("save_summary", END)

        ticker_details_generator = builder.compile()

        details = await ticker_details_generator.ainvoke({"ticker": ticker})

        summary = details.get("executive_summary", None)
        recommendation = details.get("recommendation", None)
        confidence_score = details.get("confidence_score", None)
        data_status = details.get("data_status", None)
        details = details.get("ticker_details", None)

        if details is None:
            response.status_code = 404
            return {"error": f"No data found for ticker {ticker}"}
        details: List[TickerInformation] = [
            TickerInformation(**item) for item in details
        ]
        response = ResponseData(
            ticker=ticker,
            details=[item.dict() for item in details],
            executive_summary=summary,
            recommendation=recommendation,
            confidence_score=confidence_score,
            data_status=data_status,
        )
        return response
    except Exception as e:
        response.status_code = 500
        return {"error": str(e)}
