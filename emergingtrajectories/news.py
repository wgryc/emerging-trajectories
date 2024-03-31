import requests

import feedparser

from .crawlers import crawlerPlaywright


class RSSAgent:

    def __init__(self, rss_url, crawler=None) -> None:
        """
        A simple wrapper for an RSS feed, so we can query it for URLs.

        Args:
            rss_url (str): The URL of the RSS feed.
            crawler (Crawler, optional): The crawler to use. Defaults to None, in which case we will use crawlerPlaywright in headless mode.
        """
        self.rss_url = rss_url
        if crawler is None:
            self.crawler = crawlerPlaywright()
        else:
            self.crawler = crawler

    def get_news_as_list(self) -> list:
        """
        Query the RSS feed for news articles, and return them as a list of dictionaries.

        Returns:
            list: A list of URLs.
        """
        urls = []
        feed = feedparser.parse(self.rss_url)
        for entry in feed.entries:
            urls.append(entry.link)
        return urls


class NewsAPIAgent:

    def __init__(self, api_key, top_headlines=False, crawler=None) -> None:
        """
        A simple wrapper for the News API, so we can query it for URLs.

        Args:
            api_key (str): The News API key.
            top_headlines (bool, optional): Whether to get top headlines. Defaults to False.
            crawler (Crawler, optional): The crawler to use. Defaults to None, in which case we will use crawlerPlaywright in headless mode.
        """
        self.api_key = api_key
        self.top_headlines = top_headlines
        if crawler is None:
            self.crawler = crawlerPlaywright()
        else:
            self.crawler = crawler

    def get_news_as_list(self, query: str) -> list:
        """
        Query the News API for news articles, and return them as a list of dictionaries.

        Args:
            query (str): The query to search for.

        Returns:
            list: A list of dictionaries, where each dictionary represents a news article.
        """
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={self.api_key}"
        if self.top_headlines:
            url = (
                f"https://newsapi.org/v2/top-headlines?q={query}&apiKey={self.api_key}"
            )
        response = requests.get(url)
        return response.json()
