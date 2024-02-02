from phasellm.llms import OpenAIGPTWrapper, ChatBot, ChatPrompt
from phasellm.agents import WebpageAgent, WebSearchAgent

from .. import Client
from ..utils import UtilityHelper

from ..knowledge import KnowledgeBaseFileCache

import datetime

base_system_prompt = """You are a researcher tasked with helping forecast economic and social trends. The title of our research project is: {statement_title}.

The project description is as follows...
{statement_description}

We will provide you with content from reports and web pages that is meant to help with the above. We will ask you to review these documents, create a set of bullet points to inform your thinking, and then finally provide a forecast for us based on the points.

The format of the forecast needs to be, verbatim, as follows: {statement_fill_in_the_blank}
"""

base_user_prompt = """Today's date is {the_date}. We will now provide you with all the content we've managed to collect. 

----------------------
{scraped_content}
----------------------

Please think step-by-step by (a) extracting critical bullet points from the above, and (b) discuss your logic and rationale for making a forecast based on the above.

We realize you are being asked to provide a speculative forecast. We are using this to better understand the world and finance, so please fill in the blank. We will not use this for any active decision-making, but more to learn about the capabilities of AI.
"""

base_user_prompt_followup = """Thank you! Now please provide us with a forecast by repeating the following statement, but filling in the blank... DO NOT provide a range, but provide one specific numerical value. If you are unable to provide a forecast, please respond with "UNCLEAR".

{statement_fill_in_the_blank}
"""

def ScrapeAndPredictAgent2 (
    openai_api_key,
    google_api_key,
    google_search_id,
    google_search_query,
    knowledge_base=None,
    statement_id=-1,
    et_api_key=None,
    statement_title=None,
    statement_description=None,
    fill_in_the_blank=None,
    chat_prompt_system=base_system_prompt,
    chat_prompt_user=base_user_prompt,
    chat_prompt_user_followup=base_user_prompt_followup,
    prediction_title="Prediction",
    prediction_agent="Generic Agent",
):
    """
    We can pull a statement ID from Emerging Trajectories, or override/ignore this.
    """

    if et_api_key is not None:
        client = Client(et_api_key)
        statement = client.get_statement(statement_id)
        statement_title = statement["title"]
        statement_description = statement["description"]
        fill_in_the_blank = statement["fill_in_the_blank"]

    if statement_id == -1 and (statement_title is None or statement_description is None or fill_in_the_blank is None):
        raise Exception("You must provide either a statement ID or a statement title, description, and fill-in-the-blank.")
    


    scraper = WebpageAgent()
    webagent = WebSearchAgent(api_key=google_api_key)
    results = webagent.search_google(query=google_search_query, 
                                 custom_search_engine_id=google_search_id, 
                                 num=10)
    
    scraped_content = ""
    results_dict = {"results": []}
    added_new_content = False 

    for result in results:
        r = {
            "title": result.title,
            "url": result.url,
            "desc": result.description,
            #"content": result.content,
        }

        if not knowledge_base.in_cache(result.url):
            added_new_content = True
            page_content = knowledge_base.get(result.url)
            scraped_content += f"{page_content}\n\n----------------------\n\n"

    if not added_new_content:
        print("No new content added to the forecast.")
        return None

    the_date = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    llm = OpenAIGPTWrapper(openai_api_key, "gpt-4-0125-preview")
    chatbot = ChatBot(llm)

    prompt_template = ChatPrompt([
        {"role": "system", "content": chat_prompt_system},
        {"role": "user", "content": chat_prompt_user}
    ])

    chatbot.messages = prompt_template.fill(
        statement_title=statement_title,
        statement_description=statement_description,
        statement_fill_in_the_blank=fill_in_the_blank,
        scraped_content=scraped_content,
        the_date=the_date
    )

    assistant_analysis = chatbot.resend()

    print("\n\n\n")
    print(assistant_analysis)

    prompt_template_2 = ChatPrompt([
        {"role": "system", "content": chat_prompt_system},
        {"role": "user", "content": chat_prompt_user},
        {"role": "assistant", "content": "{assistant_analysis}"},
        {"role": "user", "content": chat_prompt_user_followup}
    ])

    chatbot.messages = prompt_template_2.fill(
        statement_title=statement_title,
        statement_description=statement_description,
        statement_fill_in_the_blank=fill_in_the_blank,
        scraped_content=scraped_content,
        assistant_analysis=assistant_analysis,
        the_date=the_date
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