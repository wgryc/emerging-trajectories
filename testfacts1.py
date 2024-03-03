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

# from emergingtrajectories.knowledge import KnowledgeBaseFileCache
from emergingtrajectories.facts import FactBaseFileCache
from emergingtrajectories.recursiveagent import ETClient

client = ETClient(et_api_key)

kb = FactBaseFileCache("f_cache_test_summary_2")

statement = client.get_statement(5)
# print(statement.title)

kb.summarize_new_info(
    statement,
    chatbot,
    google_api_key,
    google_search_id,
    "Oil price projections for 2024",
    "t1.txt",
)
