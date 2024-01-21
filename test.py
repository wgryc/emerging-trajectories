import os
from dotenv import load_dotenv

load_dotenv()
et_api_key = os.getenv("ET_API_KEY")


from emergingtrajectories import Client 

c = Client(api_key = et_api_key)

#response = c.create_statement(title = "The price of Bitcoin will exceed $100,000 by the end of 2024.",
#                   description = "Same as above.",
#                   deadline = "2024-12-31T23:59:59Z")
#print(response)

#response = c.create_forecast(1, "Test", "Test", 0.1, "Agent 1")
#print(response)