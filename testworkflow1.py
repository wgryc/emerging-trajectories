#########################################################
# END-TO-END WORKFLOW FOR EMERGING TRAJECTORIES
# 2024-01-22
# By Wojciech Gryc
#########################################################

"""
This file walks you through everything you can do with the emergingtrajectories package. We begin with creating a statement and then actually generating forecasts using LLMs, including using a third-party package (phasellm) to generate forecasts with up-to-date information/data.
"""

#########################################################
# STEP 1
# Load environment variables. 
#########################################################

import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
et_api_key = os.getenv("ET_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
google_search_id = os.getenv("GOOGLE_SEARCH_ID")

#########################################################
# STEP 2
# Create a statement we'll be forecasting against.
#########################################################

from emergingtrajectories import Client 

c = Client(api_key = et_api_key)

response = c.create_statement(
    title = "Oil price prediction.",
    description = "We are predicting the price of oil for the end of the 2024 calendar year.",
    deadline = "2024-12-31T23:59:59Z")

statement_id = response["statement_id"]

#########################################################
# STEP 3
# Ask GPT-4 to generate a forecast.
#########################################################

# Note: this probably won't work as the API responses will not be comfortable making a prediction with no information. This is more of a POC to show you how we will do things with a web browsing agent.

from phasellm.llms import OpenAIGPTWrapper, ChatBot

#llm = OpenAIGPTWrapper(openai_api_key, "gpt-4-1106-preview")
llm = OpenAIGPTWrapper(openai_api_key, "gpt-3.5-turbo")
chatbot = ChatBot(llm)

step_3_system_prompt = """We are tasked with predicting the price of oil by the end of the year 2024. Could you please provide a prediction?

Think step by step and break down your prediction. I realize you might not have the relevant knowledge to make the best prediction possible, but please try anyway.

End your response by writing the following statement: 'The price of oil by end of 2024 will be at $____ USD per barrel.'"""

chatbot.messages = [
    {"role": "system", "content": step_3_system_prompt},
    {"role": "user", "content": "Please provide your prediction."}
]

prediction_text = chatbot.resend()
#print(prediction_text)

#########################################################
# STEP 4
# Use an agent to get data...
# ...then ask GPT-4 to generate a forecast.
#########################################################

from phasellm.llms import OpenAIGPTWrapper, ChatBot
from phasellm.agents import WebpageAgent, WebSearchAgent

step_4_system_prompt = """We are tasked with predicting the price of oil by the end of the year 2024. We will provide you with information we've scraped from the web to help make this prediction.

Please review all of the information we provide, and make a list of bullet points that outline your reasoning and justification for making a prediction.

Your response should end with a final paragraph that states: 'The price of oil by end of 2024 will be at $____ USD per barrel.'

The user will provide the information from various news sources."""

step_4_user_prompt = """Here is the information we have gathered about oil price projections..."""

scraper = WebpageAgent()

webagent = WebSearchAgent(api_key=google_api_key)
results = webagent.search_google(query="Oil price projections for end of 2024", 
                                 custom_search_engine_id=google_search_id, 
                                 num=10)

results_dict = {"results": []}
for result in results:
    r = {
        "title": result.title,
        "url": result.url,
        "desc": result.description,
        #"content": result.content,
    }
    page_content = result.content
    try:
        print(f"Crawling: {result.url}")
        page_content = scraper.scrape(result.url, text_only=True, body_only=True)
    except:
        print(f"Error accessing {result.url}")
    r["full_content"] = page_content
    results_dict["results"].append(r)

for result in results_dict['results']:
    step_4_user_prompt += f"\n\n----------\n\n{result['full_content']}"

step_4_user_prompt +=  f"\n\n----------\n\nNow please provide your prediction. Remember provide your thinking first, then end the prediction with the statement, 'The price of oil by end of 2024 will be at $____ USD per barrel.'"

llm = OpenAIGPTWrapper(openai_api_key, "gpt-4-1106-preview")
chatbot = ChatBot(llm)

chatbot.messages = [
    {"role": "system", "content": step_4_system_prompt},
    {"role": "user", "content": step_4_user_prompt}
]

prediction_text = chatbot.resend()
#print(prediction_text)

#########################################################
# STEP 5
# Use a utility function to extract the prediction/
#########################################################

from emergingtrajectories.utils import UtilityHelper
uh = UtilityHelper(openai_api_key)
prediction = uh.extract_prediction(
    prediction_text, 
    "The price of oil by end of 2024 will be at $____ USD per barrel. (Remove '$' if it appears, too.)"
)
#print(prediction)

#########################################################
# STEP 6
# Log the forecast.
#########################################################

response = c.create_forecast(
    statement_id, 
    "(Demo) Oil Price Prediction", 
    "This is a prediction from the demo script and workflow illustration.", 
    prediction,
    "Demo Agent", 
    {"full_response_from_llm": prediction_text}
)
print(response)
