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

from emergingtrajectories.facts import FactBaseFileCache
from emergingtrajectories.recursiveagent import ETClient
from emergingtrajectories.factsforecaster import FactForecastingAgent
from emergingtrajectories.utils import run_forecast
from emergingtrajectories import Client


def run_forecast_with_llm(factbase_loc, llm, statement_id, prediction_agent):
    etc = ETClient(et_api_key)
    fb = FactBaseFileCache(factbase_loc)
    chatbot = ChatBot(llm)
    ffa = FactForecastingAgent(etc, chatbot, fb)
    run_forecast(
        ffa.create_forecast,
        3,
        etc.get_statement(statement_id),
        openai_api_key,
        et_api_key,
        google_api_key,
        google_search_id,
        "CPI predictions for March 2024",
        prediction_agent=prediction_agent,
    )


def extend_forecast_with_llm(factbase_loc, llm, forecast_id, prediction_agent):
    etc = ETClient(et_api_key)
    fb = FactBaseFileCache(factbase_loc)
    chatbot = ChatBot(llm)
    ffa = FactForecastingAgent(etc, chatbot, fb)
    run_forecast(
        ffa.extend_forecast,
        3,
        etc.get_forecast(forecast_id),
        openai_api_key,
        et_api_key,
        google_api_key,
        google_search_id,
        "CPI predictions for March 2024",
        prediction_agent=prediction_agent,
    )


client = Client(et_api_key)

# llm = OpenAIGPTWrapper(openai_api_key, "gpt-4-0125-preview")
# run_forecast_with_llm("factbase_cpi_mar_gpt4", llm, 28, "GPT-4")
# forecast_id = client.get_most_recent_forecast(28, "GPT-4")
# extend_forecast_with_llm("factbase_cpi_mar_gpt4", llm, forecast_id, "GPT-4")

# llm = ClaudeWrapper(anthropic_api_key, "claude-2.1")
# run_forecast_with_llm("factbase_cpi_mar_claude2", llm, 28, "Claude 2.1")
# forecast_id = client.get_most_recent_forecast(28, "Claude 2.1")
# extend_forecast_with_llm("factbase_cpi_mar_claude2", llm, forecast_id, "Claude 2.1")

"""etc = ETClient(et_api_key)
fb = FactBaseFileCache("factbase_cpi_mar_claude2")
chatbot = ChatBot(llm)
ffa = FactForecastingAgent(etc, chatbot, fb)
ffa.extend_forecast(
    etc.get_forecast(forecast_id),
    openai_api_key,
    et_api_key,
    google_api_key,
    google_search_id,
    "CPI predictions for March 2024",
    prediction_agent="Claude 2.1",
)"""

# from google.cloud import aiplatform
# aiplatform.init(project="phasellm-gemini-testing")
# llm = VertexAIWrapper("gemini-1.0-pro")
# run_forecast_with_llm("factbase_cpi_mar_gemini", llm, 28, "Gemini 1.0 Pro")
# forecast_id = client.get_most_recent_forecast(28, "Gemini 1.0 Pro")
# extend_forecast_with_llm("factbase_cpi_mar_gemini", llm, forecast_id, "Gemini 1.0 Pro")

# llm = ReplicateLlama2Wrapper(replicate_api_key, "meta/llama-2-70b-chat")
# run_forecast_with_llm("factbase_cpi_mar_llama2", llm, 28, "Llama 2 70B")
# forecast_id = client.get_most_recent_forecast(28, "Llama 2 70B")
# extend_forecast_with_llm("factbase_cpi_mar_llama2", llm, forecast_id, "Llama 2 70B")

"""
llm = OpenAIGPTWrapper(openai_api_key, "gpt-4-0125-preview")
chatbot = ChatBot(llm)

etc = ETClient(et_api_key)

ffa = FactForecastingAgent(etc, chatbot, fb)"""

"""ffa.create_forecast(
    etc.get_statement(5),
    openai_api_key,
    et_api_key,
    google_api_key,
    google_search_id,
    "Oil price projections for 2024",
)"""

"""from emergingtrajectories import Client

client = Client(et_api_key)
forecast_id = client.get_most_recent_forecast(5)
print(forecast_id)

from emergingtrajectories.utils import run_forecast

run_forecast(
    ffa.extend_forecast,
    3,
    etc.get_forecast(forecast_id),
    openai_api_key,
    et_api_key,
    google_api_key,
    google_search_id,
    "Oil price projections for 2024",
)
"""
