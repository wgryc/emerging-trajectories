from emergingtrajectories.agents import ExtendScrapePredictAgent
from emergingtrajectories.knowledge import KnowledgeBaseFileCache

import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
et_api_key = os.getenv("ET_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
google_search_id = os.getenv("GOOGLE_SEARCH_ID")

kb = KnowledgeBaseFileCache("f_cache_temp_3")

agent_results = ExtendScrapePredictAgent(
    openai_api_key,
    google_api_key,
    google_search_id,
    "Boeing Share Price Projections",
    kb,
    40,
    et_api_key,
    prediction_agent="Web Scraper - Boeing (Extending Forecast)",
)

print(agent_results)
