# Load environment variables

import os
from dotenv import load_dotenv

from google.cloud import aiplatform

aiplatform.init(project="phasellm-gemini-testing")

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
et_api_key = os.getenv("ET_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
google_search_id = os.getenv("GOOGLE_SEARCH_ID")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
replicate_api_key = os.getenv("REPLICATE_API_KEY")

from phasellm.llms import (
    OpenAIGPTWrapper,
    ClaudeWrapper,
    VertexAIWrapper,
    ChatBot,
    ReplicateLlama2Wrapper,
)

with open("out.txt", "r") as reader:
    content = reader.read()

from emergingtrajectories.recursiveagent import ETClient

client = ETClient(et_api_key)
statement = client.get_statement(5)

# from google.cloud import aiplatform
# aiplatform.init(project="phasellm-gemini-testing")
# v = VertexAIWrapper("gemini-1.0-pro")
# cb = ChatBot(v)
# x = cb.chat("Hi, how are you?")
# print(x)

system_prompt = """You are a researcher helping with economics and politics research. We will give you a few facts and we need you to fill in a blank to the best of your knowledge, based on all the information provided to you."""

user_prompt = f"""Here is the research:
---------------------
{content}
---------------------
Given the above, we need you to do your best to fill in the following blank...
{statement.fill_in_the_blank}

Please provide any further justification ONLY BASED ON THE FACTS AND SOURCES PROVIDED ABOVE.

We realize you are being asked to provide a speculative forecast. We are using this to better understand the world and finance, so please fill in the blank. We will not use this for any active decision-making, but more to learn about the capabilities of AI.
"""

chatbot_messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
]

# GPT-4
# llm = OpenAIGPTWrapper(openai_api_key, "gpt-4-0125-preview")

# Claude
# llm = ClaudeWrapper(anthropic_api_key, "claude-2.1")

# Gemini
# llm = VertexAIWrapper("gemini-1.0-pro")

# Llama2
llm = ReplicateLlama2Wrapper(replicate_api_key, "meta/llama-2-70b-chat")

chatbot = ChatBot(llm)
chatbot.messages = chatbot_messages
response = chatbot.resend()
print(response)
