# This is a sample model for tracking oil prices.

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
from emergingtrajectories.factsrag import (
    FactRAGFileCache,
    FactBot,
    clean_fact_citations,
)
from emergingtrajectories.factsragforecaster import FactsRAGForecastingAgent
from emergingtrajectories import Client
from emergingtrajectories.recursiveagent import ETClient

from phasellm.llms import (
    OpenAIGPTWrapper,
    ClaudeWrapper,
    VertexAIWrapper,
    ChatBot,
    ReplicateLlama2Wrapper,
)

topic = "oil futures and oil prices for 2024"

queries = ["oil prices end of 2024 and early 2025", "oil prices today"]

crawler = crawlerPlaywright(True)
fr = FactRAGFileCache("forecasting_oil", openai_api_key, crawler=crawler)

"""
# Testing cleaning citations...

str_out = clean_fact_citations(
    fr, "Hey, this is a set of facts [f1]. This is also a set of facts [f2, f3]."
)
print(str_out)
"""

"""
# Get Content

fr.new_get_rss_links(
    "https://www.oilholicssynonymous.com/feeds/posts/default", topic=topic
)
# TODO Need to fix timeout bug.
# fr.new_get_rss_links("https://oilprice.com/rss/main", topic=topic)

fr.new_get_new_info_google(google_api_key, google_search_id, queries, topic=topic)
"""

"""
# Create a forecast

# client = Client(et_api_key)
# s = client.get_statement(5)
etc = ETClient(et_api_key)
llm = OpenAIGPTWrapper(openai_api_key, "gpt-4-0125-preview")
chatbot = ChatBot(llm)
f = FactsRAGForecastingAgent(ETClient(et_api_key), chatbot, fr)
f.create_forecast(
    etc.get_statement(5),
    openai_api_key,
    et_api_key,
    ["Today's oil price is about $85."],
    prediction_agent="FactsRAGForecastingAgent",
)
"""

"""
# Extend a forecast
etc = ETClient(et_api_key)
llm = OpenAIGPTWrapper(openai_api_key, "gpt-4-0125-preview")
chatbot = ChatBot(llm)
f = FactsRAGForecastingAgent(ETClient(et_api_key), chatbot, fr)
f.extend_forecast(
    etc.get_forecast(252),
    openai_api_key,
    et_api_key,
    prediction_agent="FactsRAGForecastingAgent",
)
"""
