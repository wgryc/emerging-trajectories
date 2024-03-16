from playwright.sync_api import sync_playwright


def run(playwright):
    browser = playwright.chromium.launch(
        headless=True
    )  # Set headless=False if you want to see the browser
    page = browser.new_page()

    # Navigate to the webpage
    page.goto("https://example.com")

    # Extract data
    # Here you can use page.query_selector() or page.content() to extract the data you need
    content = page.content()
    print(content)  # Or process the content as needed

    # Close the browser
    browser.close()


with sync_playwright() as playwright:
    run(playwright)
