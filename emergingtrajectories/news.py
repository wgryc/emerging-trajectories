import requests
import feedparser
import time
import random

from .crawlers import crawlerPlaywright, _get_text_bs4

from playwright.sync_api import sync_playwright

from news_search_client import NewsSearchClient
from azure.core.credentials import AzureKeyCredential


def force_empty_content(rss_url: str, content, cache_function) -> None:
    """
    Force the crawler to visit every URL in the RSS feed and save it as a blank content file. We do this because some RSS feeds have a lot of old URLs we do not need to crawl, and only want to crawl the delta over some period.

    Args:
        rss_url (str): The URL of the RSS feed.
        content: the content string to save.
        cache_function: the specific function to call the rss_url and content to save.
    """

    agent = RSSAgent(rss_url)
    all_urls = agent.get_news_as_list()
    for u in all_urls:
        cache_function(u, content)


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


class NewsBingAgent:

    def __init__(self, api_key: str, endpoint: str):
        """
        Creates a new Bing News API agent. To learn more, see: https://github.com/microsoft/bing-search-sdk-for-python/

        Args:
            api_key (str): The Bing News API key.
            endpoint (str): The Bing News API endpoint.
        """
        self.api_key = api_key
        self.endpoint = endpoint

    def get_news_as_list(self, query: str, market: str = "en-us") -> list:
        """
        Gets a list of URLS from the Bing News API. For more information on markets, see: https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/reference/market-codes

        Args:
            query (str): The query to search for.
            market (str, optional): The market to search in. Defaults to "en-us". (US English

        Returns:
            list: A list of URLs.
        """

        client = NewsSearchClient(
            endpoint=self.endpoint, credential=AzureKeyCredential(self.api_key)
        )

        urls = []

        try:
            news_result = client.news.search(query=query, market=market, count=10)
            for n in news_result.value:
                urls.append(n.url)

        except Exception as err:
            print("Encountered exception. {}".format(err))

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


class FinancialTimesAgent:

    # The RSS feed URLs for the Financial Times.
    ft_rss_feed_urls = [
        "https://www.ft.com/rss/home",
        "https://www.ft.com/world?format=rss",
        "https://www.ft.com/global-economy?format=rss",
        "https://www.ft.com/companies?format=rss",
        "https://www.ft.com/opinion?format=rss",
    ]

    ft_login_url = "https://ft.com/login"
    ft_main_url = "https://ft.com/"

    def __init__(self, user_email, user_password) -> None:
        """
        This is a POC agent that uses Playwright to crawl the Financial Times articles you are interested in. Note that you *NEED* to be a subscriber to the FT to make this work, and thus need to provide your FT user name and password.

        Args:
            user_email (str): Your FT email.
            user_password (str): Your FT password.
        """
        self.user_email = user_email
        self.user_password = user_password

    def get_news(self, urls: list[str] = None) -> list:
        """
        Get the news from the Financial Times as a list of tuples, where each tuple contains the URL and the extracted text content.

        Args:
            urls: a list of URLs to get content for.

        Returns:
            A list of lists -- urls, html, and text content
        """

        if urls is None:
            urls = set()
            for rss_url in self.ft_rss_feed_urls:
                agent = RSSAgent(rss_url)
                rss_url_list = agent.get_news_as_list()
                for r in rss_url_list:
                    urls.add(r)
            urls = list(urls)

        html_content_array = []
        text_content_array = []

        with sync_playwright() as playwright:

            browser = playwright.firefox.launch(headless=False)
            page = browser.new_page()

            # Navigate to the webpage
            page.goto(self.ft_main_url)

            print("Accepting Cookies")
            page.frame_locator('*[title="SP Consent Message"]').get_by_text(
                "Accept Cookies"
            ).click()

            time.sleep(2)

            page.goto(self.ft_login_url)

            time.sleep(2)

            print("Entering user name + hitting enter")

            page.locator("#enter-email").fill(self.user_email)
            page.keyboard.press("Enter")

            time.sleep(5)

            page.locator("#enter-password").fill(self.user_password)
            page.keyboard.press("Enter")

            time.sleep(5)

            url_ctr = 1
            for url in urls:
                print(f"Getting content for URL {url_ctr} of {len(urls)}")

                html_content = ""
                text_content = ""

                try:
                    page.goto(url)
                    html_content = page.content()
                    text_content = _get_text_bs4(html_content)

                    print(url)
                    print(text_content)

                except:
                    print(url)
                    print(f"Error getting content for URL {url_ctr} of {len(urls)}")

                html_content_array.append(html_content)
                text_content_array.append(text_content)

                url_ctr += 1

                time.sleep(2 + random.randint(0, 5))

            # Close the browser
            browser.close()

        return urls, html_content_array, text_content_array
