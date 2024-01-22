import requests
import json

def hello():
    print("Welcome to the Emerging Trajectories package! We've been expecting you. 😉")

class Client(object):

    base_url = "https://emergingtrajectories.com/a/api/v0.2/"

    def __init__(self, api_key):
        self.api_key = api_key

    def create_statement(self, title, description, deadline):
        url = self.base_url + "create_statement"
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        data = {
            "title": title,
            "description": description,
            "deadline": deadline
        }
        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 201:
            return response.json()
        else:
            raise Exception(response.text)
        
    def create_forecast(self, statement_id, title, justification, value, prediction_agent, additional_data={}):
        url = self.base_url + "create_forecast/" + str(statement_id)
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        data = {
            "title": title,
            "justification": justification,
            "value": value,
            "prediction_agent": prediction_agent,
            "additional_data": additional_data
        }
        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 201:
            return response.json()
        else:
            raise Exception(response.text)