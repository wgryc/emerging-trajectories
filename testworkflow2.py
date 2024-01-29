from emergingtrajectories.agents import ScrapeAndPredictAgent

import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
et_api_key = os.getenv("ET_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
google_search_id = os.getenv("GOOGLE_SEARCH_ID")

agent_results = ScrapeAndPredictAgent(
    openai_api_key,
    google_api_key,
    google_search_id,
    "Oil price projections for end of 2024",
    5,
    et_api_key,
    prediction_agent="GPT-4 with web scraping"
)

print(agent_results)