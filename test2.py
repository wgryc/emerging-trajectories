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


from emergingtrajectories import Client

client = Client(et_api_key)
# client.base_url = "http://localhost:1337/a/api/v0.2/"

forecast = client.get_most_recent_forecast(28, "Gemini 1.0 Pro")
print(forecast)

forecast = client.get_most_recent_forecast(28)
print(forecast)
