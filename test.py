import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
et_api_key = os.getenv("ET_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
google_search_id = os.getenv("GOOGLE_SEARCH_ID")

from emergingtrajectories.knowledge import statement_to_search_queries
from emergingtrajectories import Client

client = Client(et_api_key)
statement_to_search_queries(17, client, openai_api_key)
