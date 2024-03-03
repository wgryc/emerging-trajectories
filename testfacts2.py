# Load environment variables

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

from phasellm.llms import (
    OpenAIGPTWrapper,
    ClaudeWrapper,
    VertexAIWrapper,
    ChatBot,
    ReplicateLlama2Wrapper,
)

# Fact Base

from emergingtrajectories.facts import FactBaseFileCache

fb = FactBaseFileCache("f_cache_test1")

# Forecast

from emergingtrajectories.recursiveagent import ETClient
from emergingtrajectories.factsforecaster import FactForecastingAgent
from phasellm.llms import ChatBot, OpenAIGPTWrapper

llm = OpenAIGPTWrapper(openai_api_key, "gpt-4-0125-preview")
chatbot = ChatBot(llm)

etc = ETClient(et_api_key)

ffa = FactForecastingAgent(etc, chatbot, fb)

"""ffa.create_forecast(
    etc.get_statement(5),
    openai_api_key,
    et_api_key,
    google_api_key,
    google_search_id,
    "Oil price projections for 2024",
)"""

from emergingtrajectories import Client

client = Client(et_api_key)
forecast_id = client.get_most_recent_forecast(5)
print(forecast_id)

ffa.extend_forecast(
    etc.get_forecast(forecast_id),
    openai_api_key,
    et_api_key,
    google_api_key,
    google_search_id,
    "Oil price projections for 2024",
)
