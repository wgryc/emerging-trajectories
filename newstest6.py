# Testing RSS feeds
# https://www.oilholicssynonymous.com/feeds/posts/default

import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
et_api_key = os.getenv("ET_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
google_search_id = os.getenv("GOOGLE_SEARCH_ID")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
replicate_api_key = os.getenv("REPLICATE_API_KEY")
news_api_key = os.getenv("NEWS_API_KEY")

from emergingtrajectories.news import RSSAgent
from emergingtrajectories.crawlers import crawlerPlaywright
from emergingtrajectories.factsrag import FactRAGFileCache

topic = "oil futures and oil prices"

crawler = crawlerPlaywright(False)
fr = FactRAGFileCache("test_rss_oil", openai_api_key, crawler=crawler)

fr.new_get_rss_links(
    "https://www.oilholicssynonymous.com/feeds/posts/default", topic=topic
)

print(fr.query_to_fact_content("How are oil prices doing in March 2024?"))
