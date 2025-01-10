import requests
import json
from datetime import datetime
import dateparser
from typing import Union, List
import warnings
import time


def hello() -> None:
    """
    Just a hello() message/function to confirm you've installed everything!
    """
    print("Welcome to the Emerging Trajectories package! We've been expecting you. 😉")


# TODO: document
class Statement(object):
    def __init__(self, title, fill_in_the_blank):
        self.id = -1
        self.title = title
        self.fill_in_the_blank = fill_in_the_blank
        self.description = ""
        self.deadline = None
        self.created_at = None
        self.updated_at = None
        self.created_by = None


# TODO: document
class Forecast(object):

    def __init__(self, title, value, justification):
        self.id = -1
        self.title = title
        self.value = value
        self.justification = justification

        self.statement = None
        self.created_at = None
        self.updated_at = None
        self.created_by = None
        self.prediction_agent = None
        self.additional_data = {}
        self.prior_forecast = None
        self.next_forecasts = []
        self.is_human = False


class Client(object):

    # The base URL for the API, in case we need to change it or if someone wants to self-host anything.
    base_url = "https://emergingtrajectories.com/a/api/v0.2/"

    def __init__(self, api_key: str) -> None:
        """
        Launch the Emerging Trajectories Client.

        Args:
            api_key: the API key for the Emerging Trajectories platform.
        """
        self.api_key = api_key

    def create_statement(
        self, title: str, description: str, deadline: datetime, fill_in_the_blank: str
    ) -> dict:
        """
        Create a new statement that users will be forecasting against/for.

        Args:
            title: the title of the statement
            description: a more detailed description of the statement
            deadline: the deadline (date and time), typically when we'll learn the "right answer" for the forecasting process
            fill_in_the_blank: the "fill in the blank" part of the statement, which is what users will be forecasting against

        Returns:
            dict: the newly created statement returned from the platform
        """
        url = self.base_url + "create_statement"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "title": title,
            "description": description,
            "deadline": deadline,
            "fill_in_the_blank": fill_in_the_blank,
        }
        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 201:
            return response.json()
        else:
            raise Exception(response.text)

    def get_statement(self, statement_id: int) -> dict:
        """
        Returns a given statement from the platform. Includes title, description, deadline, and fill-in-the-blank.

        Args:
            statement_id: the ID of the statement to retrieve

        Returns:
            dict: the statement from the platform
        """
        url = self.base_url + "get_statement" + "/" + str(statement_id)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(response.text)

    def get_most_recent_forecast(
        self, statement_id: int, prediction_agent: str = None
    ) -> int:
        """
        Returns the most recent forecast for a given statement. This is useful for creating a new forecast that is an extension of a prior forecast.

        Args:
            statement_id: the ID of the statement to retrieve the most recent forecast for
            prediction_agent: the string for a prediction agent, if you want to further filter the most recent forecast

        Returns:
            int: the ID of the most recent forecast for the given statement
        """
        url = self.base_url + "get_most_recent_forecast/" + str(statement_id)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {}
        if prediction_agent is not None:
            data["prediction_agent"] = prediction_agent
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return int(response.json()["forecast_id"])
        else:
            raise Exception(response.text)

    def get_forecast(self, forecast_id: int) -> dict:
        """
        Returns a specific forecast's details from the platform. This typically includes the forecast title, the value associated with the fill-in-the-blank component of a statement, and justificaiton for the forecast.

        Args:
            forecast_id: the ID of the forecast to retrieve

        Returns:
            dict: the forecast details in the form of a dictionary object
        """
        url = self.base_url + "get_forecast" + "/" + str(forecast_id)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(response.text)

    def create_forecast(
        self,
        statement_id: int,
        title: str,
        justification: str,
        value: float,
        prediction_agent: str,
        additional_data: dict = {},
        prior_forecast: int = None,
        is_human: bool = False,
    ) -> None:
        """
        Creates a forecast tied to a specific statement.

        Args:
            statement_id: the ID of the statement to tie the forecast to
            title: the title of the forecast
            justification: the justification for the forecast
            value: the value associated with the fill-in-the-blank component of the statement
            prediction_agent: the agent making the prediction
            additional_data: any additional data to include with the forecast. This is not used anywhere, but can be helpful in audting or researching forecast effectiveness
            prior_forecast: if this forecast is an extension of an earlier forecast, the ID of the prior forecast
            is_human: whether the prediction is human-generated

        Returns:
            dict: the newly created forecast from the platform
        """
        url = self.base_url + "create_forecast/" + str(statement_id)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "title": title,
            "justification": justification,
            "value": value,
            "prediction_agent": prediction_agent,
            "additional_data": additional_data,
            "is_human": is_human,
        }
        if prior_forecast is not None:
            data["prior_forecast"] = prior_forecast
        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 201:
            return response.json()
        else:
            raise Exception(response.text)


class EmergingTrajectoriesClient(object):
    """
    This is the new and improved Emerging Trajectories client and should replace older clients. We keep the older ones for backwards compatability, for now.

    This client enables the same features as earlier clients, but also provides support for creating fact bases, automations, documents, and more.
    """

    # The base URL for the API, in case we need to change it or if someone wants to self-host anything.
    base_url = "https://emergingtrajectories.com/a/api/v0.2/"

    def __init__(self, api_key: str) -> None:
        """
        Args:
            api_key: the API key for the Emerging Trajectories platform.
        """
        self.api_key = api_key

    def create_factbase(self, title: str, description: str) -> str:
        """
        Create a new factbase on the Emerging Trajectories platform.

        Args:
            title: the title of the factbase
            description: a description of the factbase

        Returns:
            str: The short_code for the factbase.
        """
        url = self.base_url + "create_factbase"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "title": title,
            "description": description,
        }
        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            r = response.json()
            if "short_code" in r:
                return r["short_code"]
        else:
            raise Exception(response.text)

    def query_factbase(self, short_code, query, days_filter=1, pubdate_days_filter=-1, llm_model=None, llm_temperature=None) -> str:
        """
        Query the fact base as if you'd query a document, but with no document required. The responses here are ephemeral and are not stored anywhere.

        Args:
            short_code: the short code for the factbase to query
            query: the query to run
            days_filter: limit facts to the recent number of days
            pubdate_days_filter: limit facts to the recent number of days based on publication date
            llm_model: the LLM model to use
            llm_temperature: the LLM temperature to use
        Returns:
            str: The response written based on facts in the fact base.
        """

        url = self.base_url + "factbase_query/" + short_code 
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {"query": query}

        data['days_filter'] = days_filter
        data['pubdate_days_filter'] = pubdate_days_filter

        if llm_model is not None:
            data['llm_model'] = llm_model

        if llm_temperature is not None:
            data['llm_temperature'] = llm_temperature

        response = requests.post(url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            r = response.json()
            return r['response']
        else:
            raise Exception(response.text)

    def build_research_plan(self, query: str) -> dict:
        """
        Given a research question/query, we create a set of tasks that we can then automate document creation around.

        Args:
            query: the research question/query to build tasks around
        Returns:
            dict: a dictionary with two keys -- "text" which is the text plan, and "plan" which is a JSON plan
        """

        url = self.base_url + "api_research_analyst_build_plan"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {"query": query}

        response = requests.post(url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            r = response.json()
            return r
        else:
            raise Exception(response.text)

    def run_research_sub_task(
        self,
        command: str,
        args: Union[str, List[str]],
        document_id: int,
        short_code: str,
        wait_until_completion: bool = True,
    ) -> int:
        """
        Run the research sub-task outlined in a research plan.

        Args:
            command: the command to run
            args: the arguments to pass to the command
            document_id: the ID of the document to attach the research task to
            short_code: the short code for the factbase to use
            wait_until_completion: whether to wait until the task is completed before returning
        Returns:
            int: the ID of the job or -1 if not applicable
        """

        job_id = -1
        job_status = ""

        if command == "NEWS":
            job_id = self.run_data_collector_news(short_code, args)
        elif command == "SEARCH" or command == "WEBSEARCH":
            job_id = self.run_data_collector_serp(short_code, args, 5)
        elif (
            command == "QUERY" or command == "LIST"
        ):  # We should treat lists differently later.
            job_id = self.research_task_header_and_block(document_id, args)

        if wait_until_completion:
            job_status = ""
            while job_status != "complete":
                job_status = self.get_data_collector_job_status(job_id)
                print(job_status)
                time.sleep(10)

        return -1

    # TODO: refactor all run_data_collector_*
    def research_task_header_and_block(self, doc_id: int, query: str) -> int:
        """
        Runs a research analyst task where we generate a header block and then run the query for the document.

        Args:
            doc_id: the ID of the document to attach the research task to
            query: the query to run
        Returns:
            int: the ID of the job
        """

        url = self.base_url + "api_data_collector_run_header_and_block/" + str(doc_id)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {"query": query}

        response = requests.post(url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            r = response.json()
            if "job_id" in r:
                return int(r["job_id"])
        else:
            raise Exception(response.text)

        raise Exception("Failed to create job.")

    def run_data_collector_job(self, factbase_shortcode: str, settings: dict) -> int:
        """
        Creates a new data collector job and runs it. Contact us for information on how to pass settings.

        Args:
            factbase_shortcode: the short code for the factbase
            settings: a dictionary of settings for the data collector job
        Returns:
            int: The ID of the job
        """

        url = self.base_url + "api_data_collector_run_once/" + factbase_shortcode
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {"data_collector_settings": settings}

        response = requests.post(url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            r = response.json()
            if "job_id" in r:
                return int(r["job_id"])
        else:
            raise Exception(response.text)

        raise Exception("Failed to create job.")

    # TODO: refactor all run_data_collector_*
    def run_data_collector_serp(
        self,
        factbase_shortcode: str,
        query: Union[str, List[str]],
        n: int = 5,
        settings: dict = None,
    ) -> int:
        """
        Creates a new data collector job and runs it. Contact us for information on how to pass settings.

        Args:
            factbase_shortcode: the short code for the factbase
            query: a string to search or an array of strings
            settings: a dictionary of settings for the data collector job
        Returns:
            int: The ID of the job
        """

        if settings is None:
            settings = {}
        settings_clean = settings.copy()
        settings_clean["collector"] = "SEARCH"
        if query is type(str):
            settings_clean["query"] = [query]
        else:
            settings_clean["query"] = query

        if n > 10:
            settings_clean["n"] = 10
            warnings.warn("The maximum number of results is 10.")
        else:
            settings_clean["n"] = n

        url = self.base_url + "api_data_collector_run_serp/" + factbase_shortcode
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {"data_collector_settings": settings_clean}

        response = requests.post(url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            r = response.json()
            if "job_id" in r:
                return int(r["job_id"])
        else:
            raise Exception(response.text)

        raise Exception("Failed to create job.")

    # TODO: refactor all run_data_collector_*
    def run_data_collector_news(
        self,
        factbase_shortcode: str,
        query: Union[str, List[str]],
        n: int = 5,
        settings: dict = None,
    ) -> int:
        """
        Creates a new data collector job and runs it. Contact us for information on how to pass settings.

        Args:
            factbase_shortcode: the short code for the factbase
            query: a string to search or an array of strings
            settings: a dictionary of settings for the data collector job
        Returns:
            int: The ID of the job
        """

        if settings is None:
            settings = {}
        settings_clean = settings.copy()
        settings_clean["collector"] = "NEWSSEARCH4"
        if query is type(str):
            settings_clean["query"] = [query]
        else:
            settings_clean["query"] = query

        if n > 10:
            settings_clean["n"] = 10
            warnings.warn("The maximum number of results is 10.")
        else:
            settings_clean["n"] = n

        url = self.base_url + "api_data_collector_run_news/" + factbase_shortcode
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {"data_collector_settings": settings_clean}

        response = requests.post(url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            r = response.json()
            if "job_id" in r:
                return int(r["job_id"])
        else:
            raise Exception(response.text)

        raise Exception("Failed to create job.")

    def get_data_collector_job_status(self, job_id: int) -> str:
        """
        Get the status of a data collector job.

        Args:
            job_id: the ID of the job
        Returns:
            str: The status of the job
        """

        if job_id <= 0 or job_id is None or type(job_id) is not int:
            return (
                "complete"  # We don't have a job ID, so we'll just say it's complete.
            )

        url = self.base_url + "api_data_collector_run_once_status/" + str(job_id)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(url, headers=headers)

        if response.status_code == 200:
            r = response.json()
            if "job_status" in r:
                return r["job_status"]
        else:
            raise Exception(response.text)

        raise Exception("Failed; unknown error.")

    def create_document(
        self,
        factbase_shortcode: str,
        title: str = None,
        days_filter=99999,
        temperature=None,
        llm=None,
        facts_min_date=None,
        facts_max_date=None,
    ) -> int:
        """
        Create a new document on the Emerging Trajectories platform.

        Args:
            factbase_shortcode: the short code for the factbase to attach the document to
            title (optional): the title of the document
            days_filter (optional): limit facts to the recent number of days

        Returns:
            int: The ID of the document.
        """

        url = self.base_url + "api_doc_create"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "title": title,
            "factbase_shortcode": factbase_shortcode,
            "days_filter": days_filter,
        }

        if temperature is not None:
            data["temperature"] = temperature

        if llm is not None:
            data["llm"] = llm

        if facts_max_date is not None:
            data["facts_max_date"] = facts_max_date.isoformat()

        if facts_min_date is not None:
            data["facts_min_date"] = facts_min_date.isoformat()

        response = requests.post(url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            r = response.json()
            if "doc_id" in r:
                return int(r["doc_id"])
        else:
            raise Exception(response.text)

        raise Exception("Failed to create document.")

    def convert_facts_in_text(self, factbase_shortcode: str, text: str):
        """
        Convert text with facts to HTML code with associated links.

        Args:
            factbase_shortcode: the short code for the factbase to attach the document to
            text: the text to convert

        Returns:
            str: the converted text as HTML
            array: a list facts found in the text, with the key being the source ID in the HTML
        """

        url = self.base_url + "api_convert_facts_in_text"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {"factbase_shortcode": factbase_shortcode, "text": text}

        response = requests.post(url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            r = response.json()
            return r["html"], r["sources"]
        else:
            raise Exception(response.text)

    def update_document(
        self,
        doc_id: int,
        title: str = None,
        is_public: bool = None,
        short_code: str = None,
    ) -> bool:
        """
        Update document settings.

        Args:
            doc_id: the ID of the document
            title: the new title of the document
            is_public: whether the document is public (updated setting)
            short_code: the short code for the document (updated setting)

        Returns:
            bool: True if successful, False otherwise
        """

        url = self.base_url + "api_doc_update"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {"doc_id": doc_id}

        if title is not None:
            data["title"] = title
        if is_public is not None:
            data["is_public"] = is_public
        if short_code is not None:
            data["short_code"] = short_code

        response = requests.post(url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            return True
        else:
            raise Exception(response.text)

    def get_document(self, doc_id: int) -> dict:
        """
        Get document data.

        Args:
            doc_id: the ID of the document

        Returns:
            int: The ID of the document.
        """

        url = self.base_url + "api_doc_get"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {"doc_id": doc_id}

        response = requests.post(url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            r = response.json()
            return r
        else:
            raise Exception(response.text)

    def add_viewer(self, doc_id: int, viewer_email: str) -> bool:
        """
        Add a viewer to a private document.

        Args:
            doc_id: the ID of the document
            viewer_email: the email of the viewer to add
        Returns:
            bool: True if successful, False otherwise
        """

        url = self.base_url + "api_doc_add_viewer"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {"doc_id": doc_id, "viewer": viewer_email}

        response = requests.post(url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            return True
        else:
            raise Exception(response.text)

    def remove_viewer(self, doc_id: int, viewer_email: str) -> bool:
        """
        Remove a viewer from a private document.

        Args:
            doc_id: the ID of the document
            viewer_email: the email of the viewer to remove
        Returns:
            bool: True if successful or if the email is NOT a viewer already, False otherwise
        """

        url = self.base_url + "api_doc_remove_viewer"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {"doc_id": doc_id, "viewer": viewer_email}

        response = requests.post(url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            return True
        else:
            raise Exception(response.text)

    def get_block(self, doc_id: int, block_named_id: str) -> str:
        """
        Get the content of a named block from a document.

        Args:
            doc_id: the ID of the document
            block_named_id: the named ID of the block to retrieve
        Returns:
            str: the content of the block
        """

        url = self.base_url + "api_block_get"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {"doc_id": doc_id, "block_named_id": block_named_id}

        response = requests.post(url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            r = response.json()["content"]
            return r
        else:
            raise Exception(response.text)

    def append_header_block(self, doc_id, text: str, is_hidden: bool = False) -> bool:
        """
        Add a header block to a document.

        Args:
            doc_id: the ID of the document to append the header block to
            text: the text (header) to append
            is_hidden: whether the block is hidden in public documents
        Returns:
            bool: True if successful, False otherwise
        """

        url = self.base_url + "api_doc_append_header_block"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "doc_id": doc_id,
            "text": text,
        }

        if is_hidden:
            data["is_hidden"] = "true"

        response = requests.post(url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            return True

        return False

    def append_text_block(self, doc_id: int, text: str) -> bool:
        """
        Add a text block to a document.

        Args:
            doc_id: the ID of the document to append the text block to
            text: the text to append
        Returns:
            bool: True if successful, False otherwise
        """

        url = self.base_url + "api_doc_append_text_block"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "doc_id": doc_id,
            "text": text,
        }

        response = requests.post(url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            return True

        return False

    def append_hidden_block(self, doc_id: int, prompt: str) -> bool:
        """
        Add a text block to a document.

        Args:
            doc_id: the ID of the document to append the text block to
            prompt: the text (typically a prompt) to append
        Returns:
            bool: True if successful, False otherwise
        """

        url = self.base_url + "api_doc_append_hidden_block"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "doc_id": doc_id,
            "text": prompt,
        }

        response = requests.post(url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            return True

        return False

    def append_askai_fts_block(
        self,
        doc_id: int,
        query: str,
        fts_term: str,
        named_id: str = None,
        is_hidden: bool = False,
    ) -> dict:
        """
        Add an AI (FTS) block to a document.

        Args:
            doc_id: the ID of the document to append the text block to
            query: the query to use for the AI block
            fts_term: the full text search (FTS) terms/query to use for the AI block
            named_id: the named ID of the AI block
            is_hidden: whether the block is hidden in public documents
        Returns:
            JSON dict with document information
        """
        url = self.base_url + "api_doc_append_askai_fts_block"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "doc_id": doc_id,
            "query": query,
            "arg": fts_term,
        }

        if is_hidden:
            data["is_hidden"] = "true"

        if named_id is not None:
            data["named_id"] = named_id

        response = requests.post(url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            r = response.json()
            return r

        raise Exception(response.text)

    def append_risk_score_doc_block(
        self, doc_id: int, query: str, named_id: str = None, is_hidden: bool = False
    ) -> dict:
        """
        Add a risk score block to a document. This risk score only evaluates based on what's in the document already; it does not query the fact base.

        Args:
            doc_id: the ID of the document to append the text block to
            query: the query to use for the AI block
            named_id: the named ID of the AI block
            is_hidden: whether the block is hidden in public documents
        Returns:
            JSON dict with document information
        """
        url = self.base_url + "api_doc_append_risk_score_doc_block"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "doc_id": doc_id,
            "query": query,
        }

        if is_hidden:
            data["is_hidden"] = "true"

        if named_id is not None:
            data["named_id"] = named_id

        response = requests.post(url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            r = response.json()
            return r

        raise Exception(response.text)

    def append_askai_fts_cot_block(
        self,
        doc_id: int,
        query: str,
        fts_term: str,
        named_id: str = None,
        is_hidden: bool = False,
    ) -> dict:
        """
        Add an AI (FTS) block to a document.

        Args:
            doc_id: the ID of the document to append the text block to
            query: the query to use for the AI block
            fts_term: the full text search (FTS) terms/query to use for the AI block
            named_id: the named ID of the AI block
            is_hidden: whether the block is hidden in public documents
        Returns:
            JSON dict with document information
        """

        url = self.base_url + "api_doc_append_askai_fts_cot_block"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "doc_id": doc_id,
            "query": query,
            "arg": fts_term,
        }

        if is_hidden:
            data["is_hidden"] = "true"

        if named_id is not None:
            data["named_id"] = named_id

        response = requests.post(url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            r = response.json()
            return r

        raise Exception(response.text)

    def append_askai_cot_block(
        self, doc_id: int, query: str, named_id: str = None, is_hidden: bool = False
    ) -> dict:
        """
        Add an AI (COT) block to a document.

        Args:
            doc_id: the ID of the document to append the text block to
            query: the query to use for the AI block
            named_id: the named ID of the AI block
            is_hidden: whether the block is hidden in public documents
        Returns:
            JSON dict with document information
        """

        url = self.base_url + "api_doc_append_askai_cot_block"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "doc_id": doc_id,
            "query": query,
        }

        if is_hidden:
            data["is_hidden"] = "true"

        if named_id is not None:
            data["named_id"] = named_id

        response = requests.post(url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            r = response.json()
            return r

        raise Exception(response.text)

    def append_askai_cot_pubdate_block(
        self, doc_id: int, query: str, named_id: str = None, is_hidden: bool = False
    ) -> dict:
        """
        Add an AI (COT) block to a document.

        Args:
            doc_id: the ID of the document to append the text block to
            query: the query to use for the AI block
            named_id: the named ID of the AI block
            is_hidden: whether the block is hidden in public documents
        Returns:
            JSON dict with document information
        """

        url = self.base_url + "api_doc_append_askai_cot_pubdate_block"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "doc_id": doc_id,
            "query": query,
        }

        if is_hidden:
            data["is_hidden"] = "true"

        if named_id is not None:
            data["named_id"] = named_id

        response = requests.post(url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            r = response.json()
            return r

        raise Exception(response.text)

    def append_askai_block(
        self, doc_id: int, query: str, named_id: str = None, is_hidden: bool = False
    ) -> dict:
        """
        Add an AI block to a document.

        Args:
            doc_id: the ID of the document to append the text block to
            query: the query to use for the AI block
            is_hidden: whether the block is hidden in public documents
        Returns:
            JSON dict with document information
        """

        url = self.base_url + "api_doc_append_askai_block"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "doc_id": doc_id,
            "query": query,
        }

        if is_hidden:
            data["is_hidden"] = "true"

        if named_id is not None:
            data["named_id"] = named_id

        response = requests.post(url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            r = response.json()
            return r

        raise Exception(response.text)

    def append_askai_block_fact_range(
        self, doc_id: int, query: str, named_id: str = None, is_hidden: bool = False, num_results=10, num_before=7, num_after=7
    ) -> dict:
        """
        Add an AI 'fact range' block to a document. This uses an experimental form of RAG that should reduce hallucinations but also reduces the # of sources that can be queried.

        Args:
            doc_id: the ID of the document to append the text block to
            query: the query to use for the AI block
            is_hidden: whether the block is hidden in public documents
            num_results: the number of results (sources) to return when doing a deep dive
            num_before: the number of facts to include prior to the core fact (with chunking)
            num_after: the number of facts to include after the core fact (with chunking)
        Returns:
            JSON dict with document information
        """

        url = self.base_url + "api_doc_append_askai_fact_range_block"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "doc_id": doc_id,
            "query": query,
            "num_results": num_results,
            "num_before": num_before,
            "num_after": num_after
        }

        if is_hidden:
            data["is_hidden"] = "true"

        if named_id is not None:
            data["named_id"] = named_id

        response = requests.post(url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            r = response.json()
            return r

        raise Exception(response.text)

    def create_automation_factbase(
        self, short_code, job_type, arg_string=None, args=None
    ):
        """
        Create an automation for a fact base.

        Args:
            short_code: the short code for the factbase
            job_type: the type of job to run
            arg_string: a string of arguments
            args: a dictionary of arguments if the job requires more than one

        Returns:
            dict: the automation object
        """

        url = self.base_url + "create_automation_for_factbase" + "/" + short_code
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "job_type": job_type,
        }
        if arg_string != None:
            data["arg_string"] = arg_string
        if "args" != None:
            data["args"] = args
        response = requests.post(url, headers=headers, data=json.dumps(data))

        j = response.json()
        return j["info"]

    def create_automation_document(self, doc_id, job_type, arg_string=None, args=None):
        """
        Create an automation for a document.

        Args:
            doc_id: the document ID
            job_type: the type of job to run
            arg_string: a string of arguments
            args: a dictionary of arguments if the job requires more than one

        Returns:
            dict: the automation object
        """

        url = self.base_url + "create_automation_for_documente" + "/" + str(doc_id)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "job_type": job_type,
        }
        if arg_string != None:
            data["arg_string"] = arg_string
        if "args" != None:
            data["args"] = args
        response = requests.post(url, headers=headers, data=json.dumps(data))

        j = response.json()
        return j["info"]

    def get_factbase_automations(self, factbase_short_code):
        """
        Get an array of all automations that can be run on a factbase.

        Args:
            factbase_short_code: the short code for the factbase

        Returns:
            list: an array of automations that can be run on the factbase
        """
        url = self.base_url + f"get_automations/{factbase_short_code}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(url, headers=headers)
        j = response.json()
        print(j)
        return j["automations"]

    def queue_automation(self, automation_id):
        """
        Queue an automation (i.e., request the Emerging Trajectories platform to run an automation).

        Args:
            automation_id: the ID of the automation to queue

        Returns:
            bool: True if successful, False otherwise
        """
        url = self.base_url + f"queue_automation/{automation_id}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(url, headers=headers)
        if response.status_code == 200:
            return True
        return False

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

    def get_facts_from_factbase(
        self,
        fact_db_slug: str,
        fact_id: str = None,
        fact_ids: list = None,
        source_url: str = None,
    ) -> list[dict]:
        """
        Gets a list of facts from a fact base.

        Args:
            fact_db_slug: the slug of the fact database to get facts from.
            fact_id: the ID of the fact to retrieve.
            fact_ids: a list of fact IDs to retrieve.
            source_url: the URL of the source to retrieve facts from.

        Returns:
            list: a list of facts from the fact base.
        """

        api_url = self.base_url + "get_facts/" + fact_db_slug
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        j = {}
        if fact_id is not None:
            j["fact_id"] = fact_id
        if fact_ids is not None:
            j["fact_ids"] = fact_ids
        if source_url is not None:
            j["source_url"] = source_url

        response = requests.post(api_url, headers=headers, json=j)

        if response.status_code == 200 or response.status_code == 201:
            return response.json()["facts"]

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

    def add_url_to_factbase(
        self, fact_db_slug: str, url: Union[str, list[str]], topic: str = ""
    ) -> bool:
        """
        Sends URL to the Emerging Trajectories website, crawls the URL, and extracts facts from it.

        Args:
            fact_db_slug: the slug of the fact database to add the content to.
            url: the URL of the content, which we will crawl. Could also be an array.
            topic: the topic of the content.

        Returns:
            bool: True if successful, False otherwise.
        """

        api_url = self.base_url + "add_url_to_factbase/" + fact_db_slug
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        j = {
            "url": url,
            "topic": topic,
        }
        response = requests.post(api_url, headers=headers, json=j)

        if response.status_code == 200 or response.status_code == 201:
            return True
        print(response)
        return False
