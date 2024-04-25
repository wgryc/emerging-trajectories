"""
chunkers.py is used to chunk facts using different strategies. Emerging Trajectories started by chunking via GPT-4, but we can also appreciate using sentences, paragraphs, or other verbatim approaches. We'll be adding more chunkers as time goes on.

Chunkers should simply take a piece of content and chunk it into a list of facts.
"""

from phasellm.llms import OpenAIGPTWrapper, ChatBot, ChatPrompt
from phasellm.agents import WebpageAgent, WebSearchAgent

fact_system_prompt = """You are a researcher helping extract facts about {topic}, trends, and related observations. We will give you a piece of content scraped on the web. Please extract facts from this. Each fact should stand on its own, and can be several sentences long if need be. You can have as many facts as needed. For each fact, please start it as a new line with "---" as the bullet point. For example:

--- Fact 1... This is the fact.
--- Here is a second fact.
--- And a third fact.

Please do not include new lines between bullet points. Make sure you write your facts in ENGLISH. Translate any foreign language content/facts/observations into ENGLISH.

We will simply provide you with content and you will just provide facts."""


class ChunkerGPT4:

    def __init__(self, openai_api_key: str, model="gpt-4-turbo"):
        """
        Chunker based on GPT-4 reading text and providing a list of facts.

        Args:
            openai_api_key (str): The OpenAI API key.
            model (str): The OpenAI model to use. Defaults to "gpt-4-turbo".
        """
        self.openai_api_key = openai_api_key
        self.model = model

    def chunk(self, content: str, topic: str) -> list[str]:
        """
        Chunk text into facts.

        Args:
            content (str): The content to chunk.
            topic (str): The topic to focus on when building facts.

        Returns:
            list[str]: The list of facts.
        """

        llm = OpenAIGPTWrapper(self.openai_api_key, model=self.model)
        chatbot = ChatBot(llm)
        chatbot.messages = [{"role": "system", "content": fact_system_prompt}]

        prompt_template = ChatPrompt(
            [
                {"role": "system", "content": fact_system_prompt},
            ]
        )

        chatbot.messages = prompt_template.fill(topic=topic)

        response = chatbot.chat(content)

        lines = response.split("\n")

        facts = []

        for line in lines:
            if line[0:4] == "--- ":
                fact = line[4:]
                facts.append(fact)

        return facts


class ChunkerNewLines:

    def __init__(self, min_length: int = 7):
        """
        Chunker using line breaks for content.

        Args:
            min_length (int): The minimum length (in characters) of a fact. Defaults to 7 characters.
        """
        self.min_length = min_length

    def chunk(self, content: str, topic: str = None) -> list[str]:
        """
        Chunk text into facts.

        Args:
            content (str): The content to chunk.
            topic (str): The topic to focus on when building facts. This defaults to None so we can keep the same function calls as other chunkers.

        Returns:
            list[str]: The list of facts.
        """

        lines = content.split("\n")

        facts = []

        for line in lines:
            ls = line.strip()
            if len(ls) >= self.min_length:
                facts.append(ls)

        return facts
