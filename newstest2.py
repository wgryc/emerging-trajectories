import os
from dotenv import load_dotenv

from google.cloud import aiplatform

aiplatform.init(project="phasellm-gemini-testing")

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
et_api_key = os.getenv("ET_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
google_search_id = os.getenv("GOOGLE_SEARCH_ID")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
replicate_api_key = os.getenv("REPLICATE_API_KEY")
news_api_key = os.getenv("NEWS_API_KEY")

from emergingtrajectories.news import NewsAPIAgent
from emergingtrajectories.crawlers import crawlerPlaywright
from emergingtrajectories.factsrag import FactRAGFileCache, FactBot

topic = "Liquefied Natural Gas (LNG) futures + commodity prices"
crawler = crawlerPlaywright(False)
fr = FactRAGFileCache("test_rag", openai_api_key, crawler=crawler)
bot = FactBot(fr, openai_api_key)
