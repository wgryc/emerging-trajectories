from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup


def v3childtraversal(html):

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
            new_html = new_html + v3childtraversal(content)

    return new_html


def get_text_bs_v3(html):

    new_html = "<html><body>"

    souppre = BeautifulSoup(html, "html.parser")
    soup = souppre.body

    # print(soup.contents)
    for content in soup.contents:
        contentname = ""
        if content.name is not None:
            contentname = content.name.lower()
        if contentname not in ["script", "style"]:
            # print(content)
            new_html = new_html + v3childtraversal(content)

            # if contentname in ["p"]:
            #       new_html = new_html + str(content)
            # else:
            #       new_html = new_html + v3childtraversal(content)

    newsoup = BeautifulSoup(new_html, "html.parser")
    text = newsoup.get_text()

    return text


def get_url_content(url):

    content = ""
    text = ""
    with sync_playwright() as playwright:

        browser = playwright.chromium.launch(
            headless=False
        )  # Set headless=False if you want to see the browser
        page = browser.new_page()

        # Navigate to the webpage
        page.goto(url)

        # Extract data
        # Here you can use page.query_selector() or page.content() to extract the data you need
        content = page.content()
        # print(content)  # Or process the content as needed

        # Close the browser
        browser.close()

    text = get_text_bs_v3(content)

    return content, text
