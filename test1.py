# Load environment variables

import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
et_api_key = os.getenv("ET_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
google_search_id = os.getenv("GOOGLE_SEARCH_ID")

# Set up LLMs

from phasellm.llms import ChatBot, OpenAIGPTWrapper

llm = OpenAIGPTWrapper(openai_api_key, "gpt-4-0125-preview")
chatbot = ChatBot(llm)

# Set up knowledge base, client

from emergingtrajectories.knowledge import KnowledgeBaseFileCache

kb = KnowledgeBaseFileCache("f_cache_test1")

from emergingtrajectories.recursiveagent import ETClient

client = ETClient(et_api_key)

# Now we try and make a prediction

from emergingtrajectories.recursiveagent import RecursiveForecastingAgent

# TODO: doesn't make sense that we have the search query here.
agent = RecursiveForecastingAgent(
    client,
    chatbot,
    google_api_key,
    google_search_id,
    "CPI projections for Feb 2024",
    kb,
)

# TODO: need to provide utils support some other way. Weird to send oepnai_api_key below...

statement = client.get_statement(24)
agent.create_forecast(statement, openai_api_key, et_api_key)
