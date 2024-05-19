import requests
import json
from datetime import datetime
import dateparser


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
