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
        [
            "Wheat futures price predictions March and April 2024",
            "Wheat prices and commodity trends",
            "Wheat supply and demand risks",
        ],
        prediction_agent=prediction_agent,
        facts=[
            "Today's date is March 18, 2024 and today's spot price is about 5.42 per bushel."
        ],
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
        [
            "Wheat futures price predictions March and April 2024",
            "Wheat prices and commodity trends",
            "Wheat supply and demand risks",
        ],
        prediction_agent=prediction_agent,
    )


client = Client(et_api_key)

# llm = OpenAIGPTWrapper(openai_api_key, "gpt-4-0125-preview")
# run_forecast_with_llm("factbase_commodity_wheat_gpt_4", llm, 36, "GPT-4")
# forecast_id = client.get_most_recent_forecast(36, "GPT-4")
# extend_forecast_with_llm("factbase_commodity_wheat_gpt_4", llm, forecast_id, "GPT-4")

llm = ReplicateLlama2Wrapper(replicate_api_key, "meta/llama-2-70b-chat")
run_forecast_with_llm("factbase_commodity_wheat_llama_2", llm, 36, "Llama 2 70B")
# forecast_id = client.get_most_recent_forecast(36, "Llama 2 70B")
# extend_forecast_with_llm("factbase_commodity_wheat_llama_2", llm, forecast_id, "Llama 2 70B")
