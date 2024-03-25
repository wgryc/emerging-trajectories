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
from emergingtrajectories.factsrag import FactRAGFileCache, clean_fact_citations
from emergingtrajectories.recursiveagent import ETClient

from phasellm.llms import OpenAIGPTWrapper, ChatBot

from datetime import datetime

topic = "drivers of Liquefied Natural Gas (LNG) prices, associated political/economic/military climate, and LNG futures commodity prices"

crawler = crawlerPlaywright(False)
fr = FactRAGFileCache("testing_news_lng", openai_api_key, crawler=crawler)

etc = ETClient(et_api_key)
statement = etc.get_statement(37)

llm = OpenAIGPTWrapper(openai_api_key, "gpt-4-0125-preview")
chatbot = ChatBot(llm)

google_search_queries = [
    "Economic models and LNG",
]

"""fr.new_get_new_info_google(
    google_api_key,
    google_search_id,
    google_search_queries,
    topic,
)"""

"""fr.new_get_new_info_news(
    news_api_key, topic, ["lng", "liquefied natural gas"], top_headlines=False
)"""

"""print(
    fr.query_to_fact_content(
        "What is LNG?", n_results=3, since_date=datetime(2023, 1, 1)
    )
)"""
