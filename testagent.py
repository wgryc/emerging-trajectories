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

kb = KnowledgeBaseFileCache("f_cache_temp_july")

"""agent_results = CitationScrapeAndPredictAgent(
    openai_api_key,
    google_api_key,
    google_search_id,
    "Temperature records and observations for 2024, especially July 2024",
    kb,
    8,
    et_api_key,
    prediction_agent="[test] Citation + Web Scraper - July 2024 Temperature Anomalies",
)

print(agent_results)"""

agent_results = CiteExtendScrapePredictAgent(
    openai_api_key,
    google_api_key,
    google_search_id,
    "Forecasts for 2024 temperature and weather records",
    kb,
    79,
    et_api_key,
    prediction_agent="[test, extended] Citation + Web Scraper - July 2024 Temperature Anomalies",
)

# text = clean_citations("Hi there 33 [ ], [44], how are you [21]?", {44:"abc", 21:"def"})
# print(text)
