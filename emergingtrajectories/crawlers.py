"""
Crawlers provide a standardized approach to interacting with with web pages and extracting information. We have a number of crawlers based on PhaseLLM (Python requests) and ones using Playwright (headlessly and with a front-end) to enable flexible scraping.

All scraping agents return the raw HTML content and the extracted text content.
"""

from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

from phasellm.agents import WebpageAgent

from scrapingbee import ScrapingBeeClient


def _bs4_childtraversal(html: str) -> str:
    """
    Recursively travserse the DOM to extract content.

    Args:
        html (str): HTML content

    Returns:
        str: Extracted content
    """

    if len(str(html).strip()) < 2:
        return ""

    new_html = ""

    for content in html:
        contentname = ""

        if isinstance(content, str):
            contentname = ""
        elif content.name is not None:
            contentname = content.name.lower()

        if contentname in ["p", "pre", "h1", "h2", "h3", "h4", "h5", "h6", "span"]:
            text = content.get_text()
            num_words = len(text.strip().split(" "))
            # print(num_words)
            if num_words > 7:
                new_html = new_html + content.get_text() + "\n\n"
        else:
            new_html = new_html + _bs4_childtraversal(content)

    return new_html


def _get_text_bs4(html: str) -> str:
    """
    Extract text content from HTML using BeautifulSoup.

    Args:
        html (str): HTML content

    Returns:
        str: Extracted text content
    """

    new_html = "<html><body>"

    souppre = BeautifulSoup(html, "html.parser")
    soup = souppre.body

    for content in soup.contents:
        contentname = ""
        if content.name is not None:
            contentname = content.name.lower()
        if contentname not in ["script", "style"]:
            new_html = new_html + _bs4_childtraversal(content)

    new_html = new_html + "</body></html>"

    newsoup = BeautifulSoup(new_html, "html.parser")
    text = newsoup.get_text()

    return text


class crawlerPlaywright:

    def __init__(self, headless: bool = True) -> None:
        """
        Crawler that uses Playwright to scrape web pages.

        Args:
            headless (bool, optional): Run the browser in headless mode. Defaults to True.
        """
        self.headless = headless

    def get_content(self, url: str) -> tuple[str, str]:
        """
        Gets content for a specific URL.

        Args:
            url (str): URL to scrape

        Returns:
            tuple[str, str]: Raw HTML content and extracted text content (in this order)
        """

        content = ""
        text = ""
        with sync_playwright() as playwright:

            browser = playwright.chromium.launch(headless=self.headless)
            page = browser.new_page()

            # Navigate to the webpage
            page.goto(url)

            # Extract data
            content = page.content()

            # Close the browser
            browser.close()

        text = _get_text_bs4(content)

        return content, text


class crawlerPhaseLLM:

    def __init__(self):
        """
        PhaseLLM scraper. Uses Python requests and does not execute JS.
        """
        self.scraper = WebpageAgent()

    def get_content(self, url):
        """
        Gets content for a specific URL.

        Args:
            url (str): URL to scrape

        Returns:
            tuple[str, str]: Raw HTML content and extracted text content (in this order)
        """
        content_raw = self.scraper.scrape(url, text_only=False, body_only=False)
        content_parsed = self.scraper.scrape(url, text_only=True, body_only=True)
        return content_raw, content_parsed


class crawlerScrapingBee:

    def __init__(self, api_key: str):
        """
        Crawler that uses ScrapingBee to scrape web pages.
        """
        self.client = ScrapingBeeClient(api_key=api_key)

    def get_content(self, url):
        """
        Gets content for a specific URL.

        Args:
            url (str): URL to scrape

        Returns:
            tuple[str, str]: Raw HTML content and extracted text content (in this order)
        """

        response = self.client.get(url)
        content_raw = response.content.decode("utf-8")
        content_parsed = _get_text_bs4(content_raw)
        return content_raw, content_parsed
