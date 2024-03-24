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

na = NewsAPIAgent(news_api_key)

r = na.get_news("lng")
#r = na.get_news("covid")

for result in r['articles']:
    print("")
    print(result['title'])
    print(result['url'])
    print(result['publishedAt'])
    print("\n\n")