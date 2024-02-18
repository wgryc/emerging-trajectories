from emergingtrajectories.citationagents import (
    CitationScrapeAndPredictAgent,
    CiteExtendScrapePredictAgent,
    clean_citations,
)
from emergingtrajectories.knowledge import KnowledgeBaseFileCache

import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
et_api_key = os.getenv("ET_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
google_search_id = os.getenv("GOOGLE_SEARCH_ID")

kb = KnowledgeBaseFileCache("f_cache_commercial_real_estate_q1")

agent_results = CitationScrapeAndPredictAgent(
    openai_api_key,
    google_api_key,
    google_search_id,
    "Commercial real estate delinquencies USA Q1 2024",
    kb,
    16,
    et_api_key,
    prediction_agent="Citation + Web Scraper - ET Official",
)

print(agent_results)

"""agent_results = CiteExtendScrapePredictAgent(
    openai_api_key,
    google_api_key,
    google_search_id,
     "Forecasts and Predictions for Baltic Dry Index in 2024",
    kb,
    81,
    et_api_key,
    prediction_agent="[example] Citation + Web Scraper - Baltic Dry Index",
)

print(agent_results)"""
