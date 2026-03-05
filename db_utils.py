from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey
from datetime import datetime, timedelta, timezone
import os
import json
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()
engine = create_engine(os.getenv("DATABASE_URL"), pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class QueryHistory(Base):
    __tablename__ = "query_history"
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True)
    recommendation = Column(String)
    confidence_score = Column(Float)
    executive_summary = Column(String)
    timestamp = Column(DateTime, default=datetime.now(tz=timezone.utc))

    def __init__(self, ticker, recommendation=None, confidence_score=None, executive_summary=None):
        self.ticker = ticker
        self.recommendation = recommendation
        self.confidence_score = confidence_score
        self.executive_summary = executive_summary

    @staticmethod
    def ticker_queries_in_given_interval(
        ticker, interval_minutes=int(os.getenv("QUERY_INTERVAL_MINUTES", 60))
    ):
        with SessionLocal() as db:
            cutoff_time = datetime.now(tz=timezone.utc) - timedelta(
                minutes=interval_minutes
            )
            return (
                db.query(QueryHistory)
                .filter(
                    QueryHistory.ticker == ticker,
                    QueryHistory.timestamp > cutoff_time
                )
                .order_by(QueryHistory.timestamp.desc())
                .first()
            )
        
    def save_query(self):
        with SessionLocal() as db:
            db.add(self)
            db.commit()
            db.refresh(self)
            return self.id
    
    @staticmethod
    def delete_query(id):
        with SessionLocal() as db:
            db.query(QueryHistory).filter(QueryHistory.id == id).delete()
            db.commit()


class TickerData(Base):
    __tablename__ = "ticker_data"
    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(Integer, ForeignKey("query_history.id"))
    ticker = Column(String, index=True)
    date = Column(String)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)
    change_percent = Column(Float)
    news_headlines = Column(String)
    is_volatile = Column(Boolean)
    daily_sentiment_score = Column(Float)

    def __init__(
        self,
        query_id,
        ticker,
        date,
        open,
        high,
        low,
        close,
        volume,
        change_percent,
        news_headlines,
        is_volatile,
        daily_sentiment_score,
    ):
        self.query_id = query_id
        self.ticker = ticker
        self.date = date
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.change_percent = change_percent
        self.__setitem__("news_headlines", news_headlines)
        self.is_volatile = is_volatile
        self.daily_sentiment_score = daily_sentiment_score
    
    def __getitem__(self, key):
        if key == "news_headlines":
            nh = self.news_headlines
            if nh is None:
                return []
            if isinstance(nh, str):
                try:
                    return json.loads(nh)
                except Exception:
                    return nh
            return nh
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        if key == "news_headlines":
            self.news_headlines = json.dumps(value)
        else:
            setattr(self, key, value)

    def to_dict(self):
        return {
            "ticker": self.ticker,
            "date": self.date,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "change_percent": self.change_percent,
            "news_headlines": self["news_headlines"],
            "is_volatile": self.is_volatile,
            "daily_sentiment_score": self.daily_sentiment_score,
        }

    @staticmethod
    def fetch_ticker_data_given_query_id(query_id):
        with SessionLocal() as db:
            return (
                db.query(TickerData)
                .filter(TickerData.query_id == query_id)
                .order_by(TickerData.date.desc())
                .all()
            )
    
    @staticmethod
    def save_ticker_batch_data( batch_data):
        with SessionLocal() as db:
            db.add_all(batch_data)
            db.commit()
        
    @staticmethod
    def delete_ticker_data(query_id):
        with SessionLocal() as db:
            db.query(TickerData).filter(TickerData.query_id == query_id).delete()
            db.commit()

Base.metadata.create_all(engine)