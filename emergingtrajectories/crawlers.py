from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

from phasellm.agents import WebpageAgent


def _bs4_childtraversal(html):

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


def _get_text_bs4(html):

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

    def __init__(self, headless=True):
        self.headless = headless

    def get_content(self, url):

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
        self.scraper = WebpageAgent()

    def get_content(self, url):
        content_raw = self.scraper.scrape(url, text_only=False, body_only=False)
        content_parsed = self.scraper.scrape(url, text_only=True, body_only=True)
        return content_raw, content_parsed
