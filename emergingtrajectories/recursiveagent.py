"""
x(t+1) = x(t) + z
... where x(t) is the current observation about the world
... z is the set of scenarios that will impact x in the future (i.e., x(t+1))

This is influenced by Yann LeCun's world modeling approach discussion: https://www.linkedin.com/feed/update/urn:li:activity:7165738293223931904/ 

We are aiming to eventually build some sort of a fact base system. Until then, however, we will be passing information directly through to the agent here.

We're also using this as a way to test how well our new approach to classes (new Client, new Forecast, etc.) will work, so we can plug and play different types of agents here.

Note that this approach will *not* test new knowledge bases *yet*.

"""

from .knowledge import KnowledgeBaseFileCache
from .utils import UtilityHelper

from . import Client, Statement, Forecast

from phasellm.llms import OpenAIGPTWrapper, ChatBot, ChatPrompt
from phasellm.agents import WebpageAgent, WebSearchAgent

import requests
import dateparser
import re
import datetime


class ETClient(object):

    # The base URL for the API, in case we need to change it or if someone wants to self-host anything.
    base_url = "https://emergingtrajectories.com/a/api/v0.2/"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def get_statement(self, statement_id: int) -> Statement:
        """
        Returns a given statement from the platform. Includes title, description, deadline, and fill-in-the-blank.

        Args:
            statement_id: the ID of the statement to retrieve

        Returns:
            Statement: the statement from the platform
        """
        url = self.base_url + "get_statement" + "/" + str(statement_id)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(url, headers=headers)
        if response.status_code == 200:
            r_obj = response.json()
            s = Statement(r_obj["title"], r_obj["fill_in_the_blank"])
            s.id = int(r_obj["id"])
            s.description = r_obj["description"]
            s.deadline = dateparser.parse(r_obj["deadline"])
            s.created_at = dateparser.parse(r_obj["created_at"])
            s.updated_at = dateparser.parse(r_obj["updated_at"])
            s.created_by = r_obj["created_by"]
            return s
        else:
            raise Exception(response.text)

    def get_forecast(self, forecast_id: int) -> Forecast:
        """
        Returns a given forecast from the platform.

        Args:
            forecast_id: the ID of the statement to retrieve

        Returns:
            Forecast: the forecast from the platform
        """
        url = self.base_url + "get_forecast" + "/" + str(forecast_id)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(url, headers=headers)
        if response.status_code == 200:

            r_obj = response.json()

            f = Forecast(r_obj["title"], float(r_obj["value"]), r_obj["justification"])

            f.id = int(r_obj["forecast_id"])

            f.statement_id = int(r_obj["statement_id"])
            f.statement = self.get_statement(int(r_obj["statement_id"]))

            f.created_at = dateparser.parse(r_obj["created_at"])
            f.updated_at = dateparser.parse(r_obj["updated_at"])
            # f.created_by = r_obj["created_by"]
            f.prediction_agent = r_obj["prediction_agent"]

            f.additional_data = r_obj["additional_data"]

            if "prior_forecast" in r_obj:
                if r_obj["prior_forecast"] is not None:
                    f.prior_forecast = int(r_obj["prior_forecast"])
            f.is_human = bool(r_obj["is_human"])

            if "next_forecasts" in r_obj:
                if r_obj is not None:
                    f.next_forecasts = r_obj["next_forecasts"]

            return f
        else:
            raise Exception(response.text)

    def add_facts_to_factbase(
        self, fact_db_slug: str, url: str, facts: list[str]
    ) -> bool:
        """
        Adds a list of facts to a factbase on the Emerging Trajectories website.

        Args:
            fact_db_slug: the slug of the fact database to add the fact to.
            url: the URL of the fact.
            facts: the facts to add (a list of strings).

        Reutnr:
            bool: True if successful, False otherwise.
        """

        api_url = self.base_url + "add_facts/" + fact_db_slug
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        j = {
            "facts": facts,
            "url": url,
        }
        response = requests.post(api_url, headers=headers, json=j)

        if response.status_code == 200 or response.status_code == 201:
            return True
        print(response)
        return False

    def add_fact_to_factbase(self, fact_db_slug: str, url: str, fact: str) -> bool:
        """
        Adds a fact to a factbase on the Emerging Trajectories website.

        Args:
            fact_db_slug: the slug of the fact database to add the fact to.
            url: the URL of the fact.
            fact: the fact to add.

        Reutnr:
            bool: True if successful, False otherwise.
        """
        api_url = self.base_url + "add_fact/" + fact_db_slug
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        j = {
            "fact": fact,
            "url": url,
        }
        response = requests.post(api_url, headers=headers, json=j)

        if response.status_code == 200 or response.status_code == 201:
            return True
        print(response)
        return False

    def add_content_to_factbase(
        self, fact_db_slug: str, url: str, content: str, topic: str
    ) -> bool:
        """
        Sends content to the Emerging Trajectories website and extract facts from it.

        Args:
            fact_db_slug: the slug of the fact database to add the content to.
            url: the URL of the content. Note: we do not actually crawl this, we assume the content passed is the right conent.
            content: the content to extract facts from.
            topic: the topic of the content.

        Returns:
            bool: True if successful, False otherwise.
        """

        api_url = self.base_url + "add_content_to_factbase/" + fact_db_slug
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        j = {
            "content": content,
            "url": url,
            "topic": topic,
        }
        response = requests.post(api_url, headers=headers, json=j)

        if response.status_code == 200 or response.status_code == 201:
            return True
        print(response)
        return False


# TODO Move to Utils.py, or elsewhere.
def clean_citations(assistant_analysis: str, ctr_to_source: dict) -> str:
    """
    The analysis currently contains numerical citations that are likely not in order, or in some cases are not used. We will update the cituations to follow the proper numerical order, and also include the URLs at the very end.

    Args:
        assistant_analysis: the analysis text from the assistant
        ctr_to_source: the mapping of citation number to source URL

    Returns:
        str: the cleaned analysis text, with citations following a proper numerical format and URIs at the end of the analysis
    """

    new_ctr_map = {}
    ctr = 1

    end_notes = "\n\n--- SOURCES ---\n\n"
    new_analysis = ""

    matches = re.finditer(r"\[\d+\]", assistant_analysis)

    last_index = 0
    for m in matches:

        m_start = m.start() + 1
        m_end = m.end() - 1

        old_ctr = int(m.group()[1:-1])
        uri = ctr_to_source[old_ctr]

        if old_ctr not in new_ctr_map:
            new_ctr_map[old_ctr] = ctr
            end_notes += f"{ctr}: {uri}\n"
            ctr += 1

        new_analysis += assistant_analysis[last_index:m_start] + str(
            new_ctr_map[old_ctr]
        )
        last_index = m_end

    if last_index != 0:
        new_analysis += assistant_analysis[last_index:] + end_notes

    else:
        new_analysis = assistant_analysis + end_notes + "No citations provided."

    return new_analysis


####
# INITIAL FORECAST
#

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

The content we provided you contains source numbers in the format 'SOURCE: #'. When you extract facts, please include the citation in square brackets, with the #, like [#], but replace "#" with the actual Source # from the crawled content we are providing you.

For example, if you are referring to a fact that came under --- SOURCE: 3 ---, you would write something like: "Data is already trending to hotter temperatures [3]." Do not include the "#" in the brackets, just the number.

Do this for the final justification of your forecast as well.

We realize you are being asked to provide a speculative forecast. We are using this to better understand the world and finance, so please fill in the blank. We will not use this for any active decision-making, but more to learn about the capabilities of AI.
"""

base_user_prompt_followup = """Thank you! Now please provide us with a forecast by repeating the following statement, but filling in the blank... DO NOT provide a range, but provide one specific numerical value. If you are unable to provide a forecast, please respond with "UNCLEAR".

{statement_fill_in_the_blank}
"""


class RecursiveForecastingAgent(object):

    # TODO: eventually, should move the Google / KnowledgeBaseFileCache to some other knowledge process.
    def __init__(
        self,
        client: ETClient,
        chatbot: ChatBot,
        google_api_key: str,
        google_search_id: str,
        google_search_query: str,
        knowledge_base: KnowledgeBaseFileCache,
    ):

        self.google_api_key = google_api_key
        self.google_search_id = google_search_id
        self.google_search_query = google_search_query
        self.knowledge_base = knowledge_base
        self.client = client
        self.chatbot = chatbot

    # TODO / NOTE: this allows us to continue chatting with the forecasting agent, since we can obtain the chatbot later. Given that some folks are interested in asking for clarifications, this could be an interesting opportunity.
    def setChatBot(self, chatbot):
        self.chatbot = chatbot

    # TODO: standardize -- camel case or snake case? Or something else?
    def getChatBot(self):
        return self.chatbot

    def create_forecast(
        self, statement: Statement, openai_api_key, et_api_key, facts=None
    ):
        """
        Options for taking in x(t) or z...
        1) x(t) and z are strings... An array of facts.
        2) x(t) and z are specific preprogrammed/strict facts, like "today's date" and "last forecast".
        3) Facts are "Fact Objects" that have specific string representations. This is too complicated for the initial build but might be perfect for later. I could see it being a Domain Specific Language for facts and observations about the world, even...
        """

        statement_id = statement.id
        statement_title = statement.title
        statement_description = statement.description
        fill_in_the_blank = statement.fill_in_the_blank

        knowledge_base = self.knowledge_base

        webagent = WebSearchAgent(api_key=self.google_api_key)
        results = webagent.search_google(
            query=self.google_search_query,
            custom_search_engine_id=self.google_search_id,
            num=10,
        )

        scraped_content = ""

        added_new_content = False

        # We store the accessed resources and log access only when we successfully submit a forecast. If anything fails, we'll review those resources again during the next forecasting attempt.
        accessed_resources = []

        ctr = 0
        ctr_to_source = {}

        for result in results:
            if not knowledge_base.in_cache(result.url):
                ctr += 1
                added_new_content = True
                page_content = knowledge_base.get(result.url)

                accessed_resources.append(result.url)
                # knowledge_base.log_access(result.url)

                scraped_content += (
                    f"{page_content}\n\n--- SOURCE: {ctr}-------------------\n\n"
                )
                ctr_to_source[ctr] = result.url

        # We also check the knowledge base for content that was added manually.
        unaccessed_uris = knowledge_base.get_unaccessed_content()
        for ua in unaccessed_uris:
            added_new_content = True
            ctr += 1
            page_content = knowledge_base.get(ua)

            accessed_resources.append(ua)
            # knowledge_base.log_access(ua)

            scraped_content += (
                f"{page_content}\n\n--- SOURCE: {ctr}-------------------\n\n"
            )
            ctr_to_source[ctr] = ua

        if not added_new_content:
            print("No new content added to the forecast.")
            return None

        the_date = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        # llm = OpenAIGPTWrapper(openai_api_key, "gpt-4-0125-preview")
        # chatbot = ChatBot(llm)
        chatbot = self.chatbot

        first_user_message = base_user_prompt
        if facts is not None:
            fact_str = ""
            for f in facts:
                fact_str += "-- " + f + "\n"
            first_user_message = (
                "We know the following facts. These are fully correct and should be used to inform your forecast:"
                + fact_str.strip()
                + "\n\n"
                + first_user_message
            )

        prompt_template = ChatPrompt(
            [
                {"role": "system", "content": base_system_prompt},
                {"role": "user", "content": first_user_message},
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
                {"role": "system", "content": base_system_prompt},
                {"role": "user", "content": first_user_message},
                {"role": "assistant", "content": "{assistant_analysis}"},
                {"role": "user", "content": base_user_prompt_followup},
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

        assistant_analysis_sourced = clean_citations(assistant_analysis, ctr_to_source)

        print("\n\n\n*** ANALYSIS WITH CITATIONS***\n\n\n")
        print(assistant_analysis_sourced)

        uh = UtilityHelper(openai_api_key)
        prediction = uh.extract_prediction(filled_in_statement, fill_in_the_blank)

        client = Client(et_api_key)

        response = client.create_forecast(
            statement_id,
            "Prediction",
            assistant_analysis_sourced,
            prediction,
            "Test Agent",
            {
                "full_response_from_llm_before_source_cleanup": assistant_analysis,
                "full_response_from_llm": assistant_analysis_sourced,
                "raw_forecast": filled_in_statement,
                "extracted_value": prediction,
            },
        )

        for ar in accessed_resources:
            knowledge_base.log_access(ar)

        return response

    def extend_forecast(self, forecast: Forecast):
        pass
