from .recursiveagent import ETClient
from .facts import FactBaseFileCache
from .utils import UtilityHelper
from . import Client, Statement, Forecast

from phasellm.llms import ChatBot, OpenAIGPTWrapper, ChatPrompt

from datetime import datetime

start_system_prompt = """You are a researcher helping with economics and politics research. We will give you a few facts and we need you to fill in a blank to the best of your knowledge, based on all the information provided to you."""

start_user_prompt = """Here is the research:
---------------------
{content}
---------------------
Given the above, we need you to do your best to fill in the following blank...
{fill_in_the_blank}

PLEASE DO THE FOLLOWING:
- Provide any further justification ONLY BASED ON THE FACTS AND SOURCES PROVIDED ABOVE.
- Explain your forecast and how the facts, insights, etc. support it. Do not simply state a number.
- Do not provide a range; provide ONE number.
- End your forecast with the filled-in statement: {fill_in_the_blank_2}

We realize you are being asked to provide a speculative forecast. We are using this to better understand the world and finance, so please fill in the blank. We will not use this for any active decision-making, but more to learn about the capabilities of AI.
"""

extend_user_prompt = """Here is the research:
---------------------
{content}
---------------------

In addition to the new content above, we want to UPDATE the forecast from before. Here is the earlier forecast...
---------------------
FORECAST: {earlier_forecast_value}

JUSTIFICATION:
{earlier_forecast}
---------------------

Given the above, we need you to do your best to fill in the following blank...
{fill_in_the_blank}

PLEASE DO THE FOLLOWING:
- Provide any further justification ONLY BASED ON THE FACTS AND SOURCES PROVIDED ABOVE.
- Explain your forecast and how the facts, insights, etc. support it. Do not simply state a number.
- Do not provide a range; provide ONE number.
- End your forecast with the filled-in statement: {fill_in_the_blank_2}

We realize you are being asked to provide a speculative forecast. We are using this to better understand the world and finance, so please fill in the blank. We will not use this for any active decision-making, but more to learn about the capabilities of AI.

"""


class FactForecastingAgent(object):

    # TODO: document / clean up
    def __init__(
        self,
        client: ETClient,
        chatbot: ChatBot,
        factbase: FactBaseFileCache,
    ):

        self.client = client
        self.chatbot = chatbot
        self.factbase = factbase

    # TODO / NOTE: this allows us to continue chatting with the forecasting agent, since we can obtain the chatbot later. Given that some folks are interested in asking for clarifications, this could be an interesting opportunity.
    def setChatBot(self, chatbot):
        self.chatbot = chatbot

    # TODO: standardize -- camel case or snake case? Or something else?
    def getChatBot(self):
        return self.chatbot

    # TODO: we can do much better at disaggregating all these functions. Currently just want this to work.
    def create_forecast(
        self,
        statement: Statement,
        openai_api_key,
        et_api_key,
        google_api_key,
        google_search_id,
        google_search_query,
        facts=None,  # TODO Will provide support later.
    ):

        fact_llm = OpenAIGPTWrapper(openai_api_key, "gpt-4-0125-preview")
        fact_chatbot = ChatBot(fact_llm)

        content = self.factbase.summarize_new_info(
            statement,
            fact_chatbot,
            google_api_key,
            google_search_id,
            google_search_query,
        )

        if content is None:
            print("No new content added to the forecast.")
            return None

        chatbot_messages = [
            {"role": "system", "content": start_system_prompt},
            {"role": "user", "content": start_user_prompt},
        ]

        chatbot = self.chatbot

        prompt_template = ChatPrompt(chatbot_messages)

        the_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        chatbot.messages = prompt_template.fill(
            statement_title=statement.title,
            statement_description=statement.description,
            statement_fill_in_the_blank=statement.fill_in_the_blank,
            fill_in_the_blank_2=statement.fill_in_the_blank,
            content=content,
            the_date=the_date,  # TODO make sure we use this
        )

        assistant_analysis = chatbot.resend()

        print("\n\n\n")
        print(assistant_analysis)

        uh = UtilityHelper(openai_api_key)
        prediction = uh.extract_prediction(
            assistant_analysis, statement.fill_in_the_blank
        )

        client = Client(et_api_key)

        full_content = content + "\n\n-----------------\n\n" + assistant_analysis

        response = client.create_forecast(
            statement.id,
            "Prediction",
            full_content,
            prediction,
            "Test Agent",
            {
                "full_response_from_llm_before_source_cleanup": content,
                "full_response_from_llm": assistant_analysis,
                "extracted_value": prediction,
            },
        )

        return response

    def extend_forecast(
        self,
        forecast: Forecast,
        openai_api_key,
        et_api_key,
        google_api_key,
        google_search_id,
        google_search_query,
        facts=None,  # TODO Will provide support later.
    ):

        fact_llm = OpenAIGPTWrapper(openai_api_key, "gpt-4-0125-preview")
        fact_chatbot = ChatBot(fact_llm)

        content = self.factbase.summarize_new_info(
            forecast.statement,
            fact_chatbot,
            google_api_key,
            google_search_id,
            google_search_query,
        )

        if content is None:
            print("No new content added to the forecast.")
            return None

        chatbot_messages = [
            {"role": "system", "content": start_system_prompt},
            {"role": "user", "content": extend_user_prompt},
        ]

        chatbot = self.chatbot

        prompt_template = ChatPrompt(chatbot_messages)

        the_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        chatbot.messages = prompt_template.fill(
            statement_title=forecast.statement.title,
            statement_description=forecast.statement.description,
            statement_fill_in_the_blank=forecast.statement.fill_in_the_blank,
            fill_in_the_blank_2=forecast.statement.fill_in_the_blank,
            content=content,
            the_date=the_date,  # TODO make sure we use this
            earlier_forecast_value=str(forecast.value),
            earlier_forecast=forecast.justification,
        )

        assistant_analysis = chatbot.resend()

        print("\n\n\n")
        print(assistant_analysis)

        uh = UtilityHelper(openai_api_key)
        prediction = uh.extract_prediction(
            assistant_analysis, forecast.statement.fill_in_the_blank
        )

        client = Client(et_api_key)

        full_content = content + "\n\n-----------------\n\n" + assistant_analysis

        response = client.create_forecast(
            forecast.statement.id,
            "Prediction",
            full_content,
            prediction,
            "Test Agent",
            {
                "full_response_from_llm_before_source_cleanup": content,
                "full_response_from_llm": assistant_analysis,
                "extracted_value": prediction,
            },
            forecast.id,
        )

        return response
