from emergingtrajectories.agents import ExtendScrapePredictAgent

import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
et_api_key = os.getenv("ET_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
google_search_id = os.getenv("GOOGLE_SEARCH_ID")

agent_results = ExtendScrapePredictAgent(
    openai_api_key,
    google_api_key,
    google_search_id,
    "Boeing Share Price Projections",
    18,
    et_api_key,
    prediction_agent="Web Scraper - Boeing (Extending Forecast)",
    is_human=False
)

print(agent_results)