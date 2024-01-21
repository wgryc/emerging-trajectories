import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

from emergingtrajectories.utils import UtilityHelper
import emergingtrajectories 

uh = UtilityHelper(openai_api_key)

response = uh.extract_prediction(emergingtrajectories.utils.text_1, emergingtrajectories.utils.statement_1)

print(response)