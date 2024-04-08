# See forecast1.py instead!!

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

from emergingtrajectories import Client
from emergingtrajectories.news import NewsAPIAgent
from emergingtrajectories.crawlers import crawlerPlaywright
from emergingtrajectories.factsrag import FactRAGFileCache
from emergingtrajectories.recursiveagent import ETClient
from emergingtrajectories.factsragforecaster import FactsRAGForecastingAgent

from phasellm.llms import OpenAIGPTWrapper, ChatBot

topic = "drivers of oil prices, associated political/economic/military climate, and oil futures commodity prices"

crawler = crawlerPlaywright(False)
fr = FactRAGFileCache("testing_oil", openai_api_key, crawler=crawler)

etc = ETClient(et_api_key)
statement = etc.get_statement(5)

llm = OpenAIGPTWrapper(openai_api_key, "gpt-4-0125-preview")
chatbot = ChatBot(llm)

google_search_queries = [
    "oil price projections",
    "how do oil prices work",
    "oil and geopolitics 2024",
]

# fr.new_get_new_info_google(
#    google_api_key,
#    google_search_id,
#    google_search_queries,
#    topic,
# )

forecaster = FactsRAGForecastingAgent(etc, chatbot, fr)

"""
result = forecaster.create_forecast(
    etc.get_statement(5),
    openai_api_key,
    et_api_key,
)
print(result)
"""

"""
client = Client(et_api_key)
forecast_id = client.get_most_recent_forecast(5)
result = forecaster.extend_forecast(
    etc.get_forecast(forecast_id),
    openai_api_key,
    et_api_key,
)
print(result)
"""
