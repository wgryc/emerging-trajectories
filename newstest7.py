topic = (
    "Any news related to finance, economics, government, diplomacy, or current affairs"
)

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

ft_user = os.getenv("FT_USER_NAME")
ft_pass = os.getenv("FT_PASSWORD")

from emergingtrajectories.news import FinancialTimesAgent
from emergingtrajectories.factsrag import FactRAGFileCache

# fta = FinancialTimesAgent(ft_user, ft_pass)
# a = fta.get_news()

f = FactRAGFileCache("rag_demo_ft_rss", openai_api_key)
f.get_ft_news(ft_user, ft_pass, topic)
