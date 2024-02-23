import requests
import json
from datetime import datetime


def hello() -> None:
    """
    Just a hello() message/function to confirm you've installed everything!
    """
    print("Welcome to the Emerging Trajectories package! We've been expecting you. 😉")


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

    def get_most_recent_forecast(self, statement_id: int) -> int:
        """
        Returns the most recent forecast for a given statement. This is useful for creating a new forecast that is an extension of a prior forecast.

        Args:
            statement_id: the ID of the statement to retrieve the most recent forecast for

        Returns:
            int: the ID of the most recent forecast for the given statement
        """
        url = self.base_url + "get_most_recent_forecast/" + str(statement_id)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(url, headers=headers)
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
