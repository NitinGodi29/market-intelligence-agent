import sys
from dotenv import load_dotenv
import os
import polars as pl
import httpx
from datetime import datetime, timedelta, timezone
from transformers import pipeline
import asyncio
from db_utils import QueryHistory, TickerData
from langchain_openai import ChatOpenAI
import json

load_dotenv()

pl.Config.set_tbl_cols(-1)

financial_schema = {
    "symbol": pl.Utf8,
    "date": pl.Utf8,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Int64,
    "changePercent": pl.Float64,
    "news_headlines": pl.List(pl.Utf8),
    "is_volatile": pl.Boolean,
}

pipe = pipeline("sentiment-analysis", model=os.getenv("MODEL_NAME"))
model = ChatOpenAI(
    model="gpt-5-mini-2025-08-07", temperature=0, api_key=os.getenv("OPENAI_API_KEY")
)


async def fetch_quote(ticker):
    stock_quote_url = os.getenv("FMP_STOCK_QUOTE_URL")
    query_params = {
        "symbol": ticker,
        "from": (
            datetime.now() - timedelta(days=int(os.getenv("HISTORICAL_DAYS", 1)))
        ).strftime("%Y-%m-%d"),
        "apikey": os.getenv("FMP_API_KEY"),
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(stock_quote_url, params=query_params)
        if response.status_code == 200:
            data = response.json()
            # print(f"Received data: {data}")
            return data
        else:
            # print(
            #     f"Failed to fetch stock quote for ticker {ticker}. Status code: {response.status_code}"
            # )
            return None
    except Exception as e:
        print(f"Error fetching stock quote for ticker {ticker}: {e}")
        return None


async def fetch_news(ticker):
    stock_news_url = os.getenv("FINNHUB_STOCK_NEWS_URL")
    query_params = {
        "symbol": ticker,
        "from": (
            datetime.now() - timedelta(days=int(os.getenv("HISTORICAL_DAYS", 1)))
        ).strftime("%Y-%m-%d"),
        "token": os.getenv("FINNHUB_API_KEY"),
    }
    try:
        async with httpx.AsyncClient() as client:
            news_response = await client.get(stock_news_url, params=query_params)
        if news_response.status_code == 200:
            news_data = news_response.json()
            # print(f"News data for {ticker}: {news_data}")
            return news_data
        else:
            # print(
            #     f"Failed to fetch news data for ticker {ticker}. Status code: {news_response.status_code}"
            # )
            return None
    except Exception as e:
        print(f"Error fetching news data for ticker {ticker}: {e}")
        return None


async def get_aggregated_sentiment_score(headlines):
    """Run the blocking `pipeline` in a threadpool so the event loop isn't blocked.

    Returns a float sentiment score (0.0 if no headlines).
    """
    if not list(headlines):
        return 0.0
    loop = asyncio.get_running_loop()
    try:
        # Run the blocking pipeline call in the default ThreadPoolExecutor
        sentiment_scores = await loop.run_in_executor(
            None, lambda: pipe(list(headlines))
        )
        if not sentiment_scores:
            return 0.0
        score_sum = 0
        for sentiment_score in sentiment_scores:
            if sentiment_score.get("label") == "positive":
                score_sum += 1
            elif sentiment_score.get("label") == "negative":
                score_sum -= 1
        return score_sum / len(sentiment_scores)
    except Exception as e:
        # Non-fatal: log and return None so callers can represent missing scores as nulls
        print(f"Error computing sentiment for headlines: {e}")
        return None


def transform_market_data(data):
    ldf = pl.LazyFrame(data, schema=financial_schema)
    ldf = (
        ldf.with_columns(
            (
                pl.col("changePercent").abs()
                > float(os.getenv("VOLATILITY_THRESHOLD", 5.0))
            ).alias("is_volatile")
        )
        .fill_null(True)
        .sort("date", descending=True)
        .rename({"changePercent": "change_percent", "symbol": "ticker"})
    )
    return ldf.collect()


async def get_ticker_details(state):
    ticker = state["ticker"]
    # print(f"Fetching details for ticker: {ticker}")
    existing_query = QueryHistory.ticker_queries_in_given_interval(ticker)
    if existing_query:
        existing_ticker_data = TickerData.fetch_ticker_data_given_query_id(
            existing_query.id
        )
        if existing_ticker_data:
            # print(f"Returning cached data for ticker {ticker}")
            details = [item.to_dict() for item in existing_ticker_data]
            response = {
                "ticker": ticker,
                "ticker_details": details,
                "executive_summary": existing_query.executive_summary,
                "recommendation": existing_query.recommendation,
                "confidence_score": existing_query.confidence_score,
                "data_status": "Cached data from recent query",
            }
            return response
        else:
            # print(f"No ticker data found for existing query of ticker {ticker}, deleting stale query")
            QueryHistory.delete_query(existing_query.id)

    quote_data = await fetch_quote(ticker)
    if quote_data is None:
        print(f"No quote data available for ticker {ticker}")
        return
    news_data = await fetch_news(ticker)
    if news_data is None:
        print(f"No news data available for ticker {ticker}")
        return

    for item in quote_data:
        item["news_headlines"] = []
        for ele in news_data:
            if (
                datetime.fromtimestamp(ele["datetime"], tz=timezone.utc).date()
                == datetime.strptime(item["date"], "%Y-%m-%d").date()
            ) and len(ele["headline"]) > int(os.getenv("MIN_HEADLINE_LENGTH", 10)):
                if datetime.fromtimestamp(ele["datetime"], tz=timezone.utc).strftime(
                    "%H:%M:%S"
                ) > os.getenv("NASDAQ_CLOSE_TIME", "16:00:00"):
                    continue
                item["news_headlines"].append(ele["headline"])

    result = transform_market_data(quote_data)
    # Compute daily sentiment scores concurrently on the DataFrame rows
    headlines_list = result["news_headlines"].to_list()
    tasks = [
        asyncio.create_task(get_aggregated_sentiment_score(h)) for h in headlines_list
    ]
    scores = await asyncio.gather(*tasks) if tasks else []
    # add the computed scores as a new column (or nulls when no scores)
    if scores:
        result = result.with_columns(pl.Series("daily_sentiment_score", scores))
    else:
        # create a column of nulls with the same length as the DataFrame
        nulls = [None] * result.height
        result = result.with_columns(pl.Series("daily_sentiment_score", nulls))
    # print(f"Final DataFrame for {ticker}:\n{result}")
    return {**state, "ticker_details": result.to_dicts(), "data_status": "Fresh data from sources"}

def conditional_edge(state):
    details = state.get("ticker_details", [])
    if not details:
        return "quite_market"
    for item in details:
        if item.get("is_volatile", True):
            return "generate_summary"
    return "quite_market"

def return_quite_market_response(state):
    return {
        **state,
        "executive_summary": "No significant market activity or news for this ticker in the given timeframe.",
        "recommendation": "Hold",
        "confidence_score": 0.95,
    }

def generate_summary_using_llm(state):
    details = state.get("ticker_details", [])
    todays_change = details[0].get("change_percent")
    todays_sentiment = details[0].get("daily_sentiment_score")
    average_prev_sentiment = sum(
        item.get("daily_sentiment_score", 0) for item in details[1:]
    ) / max(len(details) - 1, 1)
    prompt = f"""Given the following recent stock market data for a ticker "{state['ticker']}", generate a recommendation in terms of Buy, Sell, or Hold, along with a confidence score between 0 and 1. Also, provide a brief executive summary explaining the recommendation.
                Market Data:
                Today's Change: {todays_change}%
                Today's Sentiment Score: {todays_sentiment}
                Average Previous Sentiment Score: {average_prev_sentiment}
                Please provide the output in the following JSON format:
                {{
                "recommendation": "Buy/Sell/Hold",
                "confidence_score": 0.0-1.0,
                "executive_summary": "Brief explanation of the recommendation based on the market data."
                }}
            """
    system_message = (
        "You are a helpful and precise financial assistant for stock market analysis."
    )
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]
    # response = model.invoke(messages)
    try:
        # response_dict = json.loads(response.content)
        response_dict = {
            "recommendation": "Testing",
            "confidence_score": 0,
            "executive_summary": "Testing"}
        return {
            **state,
            "executive_summary": response_dict.get("executive_summary"),
            "recommendation": response_dict.get("recommendation"),
            "confidence_score": response_dict.get("confidence_score"),
        }
    except (json.JSONDecodeError, KeyError):
        return {
            **state,
            "executive_summary": "Unable to generate a summary based on the provided data.",
            "recommendation": "N/A",
            "confidence_score": "N/A",
        }

def save_summary_to_db(state):
    query = QueryHistory(
        ticker=state.get("ticker"),
        recommendation=state.get("recommendation"),
        confidence_score=state.get("confidence_score"),
        executive_summary=state.get("executive_summary"),
    )
    query_id = QueryHistory.save_query(query)
    ticker_details = state.get("ticker_details", [])
    TickerData.save_ticker_batch_data(
        [TickerData(query_id=query_id, **item) for item in ticker_details]
    )
if __name__ == "__main__":
    ticker = sys.argv[1]
    # print(f"Processing ticker: {ticker}")
    get_ticker_details(ticker)
