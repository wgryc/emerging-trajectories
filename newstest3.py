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
from emergingtrajectories.crawlers import crawlerPlaywright
from emergingtrajectories.factsrag import FactRAGFileCache
from emergingtrajectories.recursiveagent import ETClient

from phasellm.llms import OpenAIGPTWrapper, ChatBot

topic = "drivers of Liquefied Natural Gas (LNG) prices, associated political/economic/military climate, and LNG futures commodity prices"

crawler = crawlerPlaywright(False)
fr = FactRAGFileCache("rag_lng", openai_api_key, crawler=crawler)

etc = ETClient(et_api_key)
statement = etc.get_statement(37)

llm = OpenAIGPTWrapper(openai_api_key, "gpt-4-0125-preview")
chatbot = ChatBot(llm)

google_search_queries = [
    "LNG futures prices and estimmates",
    "LNG prices in 2024 and 2025",
    "Political, economic, and military drivers of LNG prices",
    "Biggest producers of LNG",
    "Biggest consumers of LNG",
    "LNG and Natural Gas price and structural relationships",
    "The politics of LNG",
    "LNG regulations in USA",
    "LNG regulations in China",
    "LNG regulations in Europe",
    "LNG and the Russia/Ukraine war",
    "The economics of LNG production",
    "Economic models and LNG",
]

fr.summarize_new_info_multiple_queries(
    statement,
    chatbot,
    google_api_key,
    google_search_id,
    google_search_queries,
    topic,
)
