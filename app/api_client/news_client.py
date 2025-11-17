"""
News aggregation clients for football availability monitoring.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional

import httpx
import feedparser

logger = logging.getLogger("news_client")
logger.setLevel(logging.INFO)


class BaseNewsClient:
    def fetch_articles(self) -> List[Dict[str, Any]]:
        raise NotImplementedError


class GNewsClient(BaseNewsClient):
    ENDPOINT = "https://gnews.io/api/v4/search"

    def __init__(self, api_key: str, lang: str = "en", country: Optional[str] = None) -> None:
        self.api_key = api_key
        self.lang = lang
        self.country = country
        self.session = httpx.Client(timeout=10.0)

    def fetch_articles(self, query: str = "football") -> List[Dict[str, Any]]:
        params = {
            "token": self.api_key,
            "lang": self.lang,
            "q": query,
            "max": 20,
        }
        if self.country:
            params["country"] = self.country
        try:
            resp = self.session.get(self.ENDPOINT, params=params)
            resp.raise_for_status()
            data = resp.json()
            return data.get("articles", [])
        except httpx.HTTPError as exc:
            logger.warning(f"⚠️ GNews fetch failed: {exc}")
            return []


class NewsAPIClient(BaseNewsClient):
    ENDPOINT = "https://newsapi.org/v2/top-headlines"

    def __init__(self, api_key: str, country: str = "gb", category: str = "sports") -> None:
        self.api_key = api_key
        self.country = country
        self.category = category
        self.session = httpx.Client(timeout=10.0)

    def fetch_articles(self) -> List[Dict[str, Any]]:
        params = {
            "apiKey": self.api_key,
            "country": self.country,
            "category": self.category,
            "pageSize": 20,
        }
        try:
            resp = self.session.get(self.ENDPOINT, params=params)
            resp.raise_for_status()
            data = resp.json()
            return data.get("articles", [])
        except httpx.HTTPError as exc:
            logger.warning(f"⚠️ NewsAPI fetch failed: {exc}")
            return []


class RSSClient(BaseNewsClient):
    def __init__(self, feeds: List[Dict[str, str]]) -> None:
        self.feeds = feeds

    def fetch_articles(self) -> List[Dict[str, Any]]:
        articles: List[Dict[str, Any]] = []
        for feed in self.feeds:
            try:
                parsed = feedparser.parse(feed["url"])
                for entry in parsed.entries:
                    articles.append(
                        {
                            "title": entry.get("title"),
                            "link": entry.get("link"),
                            "publishedAt": entry.get("published"),
                            "source": feed.get("name", "RSS"),
                            "description": entry.get("summary"),
                        }
                    )
            except Exception as exc:
                logger.warning(f"⚠️ RSS fetch failed for {feed.get('name')}: {exc}")
        return articles

