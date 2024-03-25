import requests

from .crawlers import crawlerPlaywright


class NewsAPIAgent:

    def __init__(self, api_key, top_headlines=False, crawler=None):
        self.api_key = api_key
        self.top_headlines = top_headlines
        if crawler is None:
            self.crawler = crawlerPlaywright()
        else:
            self.crawler = crawler

    def get_news_as_list(self, query):
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={self.api_key}"
        if self.top_headlines:
            url = (
                f"https://newsapi.org/v2/top-headlines?q={query}&apiKey={self.api_key}"
            )
        response = requests.get(url)
        return response.json()
