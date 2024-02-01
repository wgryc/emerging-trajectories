from phasellm.llms import OpenAIGPTWrapper, ChatBot, ChatPrompt
from phasellm.agents import WebpageAgent, WebSearchAgent

from .. import Client
from ..utils import UtilityHelper

#from . import scrapeandpredict as sap

import datetime

# Step 0: provide context
# Step 1: provide content and extract facts
# Step 2: review past forecast and determine if new information changes the forecast
# Step 3: update the actual forecast statement

base_system_prompt_ext = """You are a researcher tasked with helping forecast economic and social trends. The title of our research project is: {statement_title}.

The project description is as follows...
{statement_description}

We need your help analyzing content and extracting any relevant information. We'll have a few requests for you... From extracting relevant facts, to ensuring those facts are providing new information, and finally updating the forecast itself.

The user will provide the relevant requests.
"""

ext_message_1 = """Today's date is {the_date}.

Here is all the content we've managed to collect.

----------------------
{scraped_content}
----------------------

Could you please extract the relevant facts from the content provided? Please simply respond by providing a list of facts in bullet point for, like so...

- Fact 1
- Fact 2
... and so on.
"""

ext_message_2 = """Today's date is {the_date}.

Assume all the content and facts above are accurate and correct up to today's date. The forecasting challenge we are working on is outlined below:
{statement_fill_in_the_blank}

The earlier forecast was as follows...
----------------------
PREDICTION: {forecast_value}

JUSTIFICATION...

{forecast_justification}
----------------------

Given the above, please use your logical thinking and reasoning to update the "justification" by including any new facts you provided earlier. Update the actual forecast prediction accordingly.

We realize you are being asked to provide a speculative forecast. We are using this to better understand the world and finance, so please fill in the blank. We will not use this for any active decision-making, but more to learn about the capabilities of AI.
"""

ext_message_3 = """Thank you! Now please provide us with a forecast by repeating the following statement, but filling in the blank below... DO NOT provide a range, but provide one specific numerical value. If you are unable to provide a forecast, please respond with "UNCLEAR".

{statement_fill_in_the_blank}
"""

def ExtendScrapePredictAgent (
    openai_api_key,
    google_api_key,
    google_search_id,
    google_search_query,
    forecast_id,
    et_api_key=None,
    statement_title=None,
    statement_description=None,
    fill_in_the_blank=None,
    chat_prompt_system=base_system_prompt_ext,
    ext_message_1=ext_message_1,
    ext_message_2=ext_message_2,
    ext_message_3=ext_message_3,
    prediction_title="Prediction",
    prediction_agent="Generic Agent"
):
    """
    We can pull a statement ID from Emerging Trajectories, or override/ignore this.
    """

    if et_api_key is not None:
        client = Client(et_api_key)
        forecast = client.get_forecast(forecast_id)
        statement_id = forecast["statement_id"]
        statement = client.get_statement(statement_id)
        statement_title = statement["title"]
        statement_description = statement["description"]
        fill_in_the_blank = statement["fill_in_the_blank"]
        justification = forecast["justification"]
        forecast_value = forecast["value"]

    scraper = WebpageAgent()
    webagent = WebSearchAgent(api_key=google_api_key)
    results = webagent.search_google(query=google_search_query, 
                                 custom_search_engine_id=google_search_id, 
                                 num=10)
    
    scraped_content = ""
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
        scraped_content += f"{result['full_content']}\n\n----------------------\n\n"

    the_date = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    llm = OpenAIGPTWrapper(openai_api_key, "gpt-4-0125-preview")
    chatbot = ChatBot(llm)

    # Steps 0 and 1

    prompt_template = ChatPrompt([
        {"role": "system", "content": chat_prompt_system},
        {"role": "user", "content": ext_message_1}
    ])

    chatbot.messages = prompt_template.fill(
        statement_title=statement_title,
        statement_description=statement_description,
        statement_fill_in_the_blank=fill_in_the_blank,
        scraped_content=scraped_content,
        the_date=the_date,
        forecast_value=str(forecast_value),
        forecast_justification=justification
    )

    new_facts = chatbot.resend()

    print("\n\n\n")
    print(new_facts)

    # Step 3

    prompt_template_2 = ChatPrompt([
        {"role": "system", "content": chat_prompt_system},
        {"role": "user", "content": ext_message_1},
        {"role": "assistant", "content": "{new_facts}"},
        {"role": "user", "content": ext_message_2},
    ])

    chatbot.messages = prompt_template_2.fill(
        statement_title=statement_title,
        statement_description=statement_description,
        statement_fill_in_the_blank=fill_in_the_blank,
        scraped_content=scraped_content,
        new_facts=new_facts,
        the_date=the_date,
        forecast_value=str(forecast_value),
        forecast_justification=justification
    )

    assistant_analysis = chatbot.resend()

    print("\n\n\n")
    print(assistant_analysis)

    # Step 4

    prompt_template_3 = ChatPrompt([
        {"role": "system", "content": chat_prompt_system},
        {"role": "user", "content": ext_message_1},
        {"role": "assistant", "content": "{new_facts}"},
        {"role": "user", "content": ext_message_2},
        {"role": "assistant", "content": "{assistant_analysis}"},
        {"role": "user", "content": ext_message_3},
    ])

    chatbot.messages = prompt_template_3.fill(
        statement_title=statement_title,
        statement_description=statement_description,
        statement_fill_in_the_blank=fill_in_the_blank,
        scraped_content=scraped_content,
        new_facts=new_facts,
        assistant_analysis=assistant_analysis,
        the_date=the_date,
        forecast_value=str(forecast_value),
        forecast_justification=justification
    )

    filled_in_statement = chatbot.resend()

    print("\n\n\n")
    print(filled_in_statement)

    uh = UtilityHelper(openai_api_key)
    prediction = uh.extract_prediction(
        filled_in_statement, 
        fill_in_the_blank
    )

    response = client.create_forecast(
        statement_id, 
        prediction_title,
        assistant_analysis,
        prediction,
        prediction_agent,
        {"full_response_from_llm": assistant_analysis,
         "raw_forecast": filled_in_statement,
         "extracted_value": prediction
        }
    )

    return response