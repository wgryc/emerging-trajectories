from emergingtrajectories.knowledge import KnowledgeBaseFileCache

#w = KnowledgeBaseFileCache("f_cachetest")
#content = w.get("https://10millionsteps.com/ai-inflection")
#print(content)

from emergingtrajectories.agents import ScrapeAndPredictAgent2

import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
et_api_key = os.getenv("ET_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
google_search_id = os.getenv("GOOGLE_SEARCH_ID")

kb = KnowledgeBaseFileCache("f_cachetest")

agent_results = ScrapeAndPredictAgent2(
    openai_api_key,
    google_api_key,
    google_search_id,
    "Oil price projections for end of 2024",
    kb,
    5,
    et_api_key,
    prediction_agent="GPT-4 with web scraping"
)

print(agent_results)