from phasellm.llms import OpenAIGPTWrapper, ChatBot, ChatPrompt
from phasellm.agents import WebSearchAgent

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


# In this case, we also get any documents that haven't been accessed by the agent.
# This is why agent <-> kb needs to be a 1:1 relationship.
def ScrapeAndPredictAgent(
    openai_api_key: str,
    google_api_key: str,
    google_search_id: str,
    google_search_query: str,
    knowledge_base: KnowledgeBaseFileCache = None,
    statement_id: int = -1,
    et_api_key: str = None,
    statement_title: str = None,
    statement_description: str = None,
    fill_in_the_blank: str = None,
    chat_prompt_system: str = base_system_prompt,
    chat_prompt_user: str = base_user_prompt,
    chat_prompt_user_followup: str = base_user_prompt_followup,
    prediction_title: str = "Prediction",
    prediction_agent: str = "Generic Agent",
) -> dict:
    """
    This agent submits a search query to Google to find information related to its forecast. It also uses any information that it has not previously accessed in its KnowledgeBase. It then generates a forecast with all the relevant information.

    Args:
        openai_api_key: the OpenAI API key
        google_api_key: the Google Search API key
        google_search_id: the Google search ID
        google_search_query: the Google search query
        knowledge_base: the KnowledgeBaseFileCache object
        statement_id: the ID of the statement to use
        et_api_key: the Emerging Trajectories API key
        statement_title: the title of the statement (if not submitting a statement ID)
        statement_description: the description of the statement (if not submitting a statement ID)
        fill_in_the_blank: the fill-in-the-blank component of the statement (if not submitting a statement ID)
        chat_prompt_system: the system prompt for the chatbot (optional, for overriding defaults)
        chat_prompt_user: the user prompt for the chatbot (optional, for overriding defaults)
        chat_prompt_user_followup: the follow-up user prompt for the chatbot (optional, for overriding defaults)
        prediction_title: the title of the forecast
        prediction_agent: the agent making the forecast

    Returns:
        dict: the response from the Emerging Trajectories platform
    """

    if et_api_key is not None:
        client = Client(et_api_key)
        statement = client.get_statement(statement_id)
        statement_title = statement["title"]
        statement_description = statement["description"]
        fill_in_the_blank = statement["fill_in_the_blank"]

    if statement_id == -1 and (
        statement_title is None
        or statement_description is None
        or fill_in_the_blank is None
    ):
        raise Exception(
            "You must provide either a statement ID or a statement title, description, and fill-in-the-blank."
        )

    webagent = WebSearchAgent(api_key=google_api_key)
    results = webagent.search_google(
        query=google_search_query, custom_search_engine_id=google_search_id, num=10
    )

    scraped_content = ""

    added_new_content = False

    for result in results:
        if not knowledge_base.in_cache(result.url):
            added_new_content = True
            page_content = knowledge_base.get(result.url)
            knowledge_base.log_access(result.url)
            scraped_content += f"{page_content}\n\n----------------------\n\n"

    # We also check the knowledge base for content that was added manually.
    unaccessed_uris = knowledge_base.get_unaccessed_content()
    for ua in unaccessed_uris:
        added_new_content = True
        page_content = knowledge_base.get(ua)
        knowledge_base.log_access(ua)
        scraped_content += f"{page_content}\n\n----------------------\n\n"

    if not added_new_content:
        print("No new content added to the forecast.")
        return None

    the_date = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    llm = OpenAIGPTWrapper(openai_api_key, "gpt-4-0125-preview")
    chatbot = ChatBot(llm)

    prompt_template = ChatPrompt(
        [
            {"role": "system", "content": chat_prompt_system},
            {"role": "user", "content": chat_prompt_user},
        ]
    )

    chatbot.messages = prompt_template.fill(
        statement_title=statement_title,
        statement_description=statement_description,
        statement_fill_in_the_blank=fill_in_the_blank,
        scraped_content=scraped_content,
        the_date=the_date,
    )

    assistant_analysis = chatbot.resend()

    print("\n\n\n")
    print(assistant_analysis)

    prompt_template_2 = ChatPrompt(
        [
            {"role": "system", "content": chat_prompt_system},
            {"role": "user", "content": chat_prompt_user},
            {"role": "assistant", "content": "{assistant_analysis}"},
            {"role": "user", "content": chat_prompt_user_followup},
        ]
    )

    chatbot.messages = prompt_template_2.fill(
        statement_title=statement_title,
        statement_description=statement_description,
        statement_fill_in_the_blank=fill_in_the_blank,
        scraped_content=scraped_content,
        assistant_analysis=assistant_analysis,
        the_date=the_date,
    )

    filled_in_statement = chatbot.resend()

    print("\n\n\n")
    print(filled_in_statement)

    uh = UtilityHelper(openai_api_key)
    prediction = uh.extract_prediction(filled_in_statement, fill_in_the_blank)

    response = client.create_forecast(
        statement_id,
        prediction_title,
        assistant_analysis,
        prediction,
        prediction_agent,
        {
            "full_response_from_llm": assistant_analysis,
            "raw_forecast": filled_in_statement,
            "extracted_value": prediction,
        },
    )

    return response
