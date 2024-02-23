"""
Solutions for finding, extracting, storing, and reviisting knowledge.
"""

"""
This is the first knowledge base, and is meant to be a POC, really.

All of our agents as of today (Feb 1) focus on web searches and website content. Today, we do a Google search and scrape the content from the top results. We repeat this process every time the agent runs.

An obvious next step would be to create some sort of a cache to see if we already scraped the page and included the content elsewhere.

This should also be able to do multiple searches *and* accept other URLs to scrape.

How would this one work?
1. Have a folder where things get cached.
2. Have a JSON file that tracks when a knowledge base was accessed, the source URL, etc.
"""

import os
import json
import hashlib

# Using JSONEncoder to be consistent with the Emerging Trajectories website and platform.
from django.core.serializers.json import DjangoJSONEncoder

from phasellm.agents import WebpageAgent

from datetime import datetime

from . import Client
from phasellm.llms import OpenAIGPTWrapper, ChatBot

"""
CACHE STRUCTURE IN JSON...

key: URI (URL or file name)
value: {
    "obtained_on": <date>; when the file was downloaded
    "last_accessed": <date>; when the file was last used by the agent
    "accessed": 0 if not accessed, 1 if accessed
    "uri_md5": the MD5 sum of the URI
}

"""


def statement_to_search_queries(
    statement_id: int, client: Client, openai_api_key: str, num_queries: int = 3
) -> list[str]:
    """
    Given a specific statement ID, this will return a list of queries you can put into a search engine to get useful information.

    Args:
        statement_id (int): The ID of the statement to get search queries for.
        client (Client): The Emerging Trajectories API client.
        openai_api_key (str): The OpenAI API key.
        num_queries (int, optional): The number of queries to return. Defaults to 3.

    Returns:
        list[str]: A list of search queries.

    """

    statement = client.get_statement(statement_id)
    # print(statement)

    llm = OpenAIGPTWrapper(openai_api_key, model="gpt-3.5-turbo")
    chatbot = ChatBot(llm)

    chatbot.messages = [
        {
            "role": "system",
            "content": f"""I am working on a research project about this topic:\n{statement['title']}\n\n{statement['description']}\n\nHere is more information about what I am trying to do:\n{statement['description']}""",
        },
        {
            "role": "user",
            "content": "Could you please provide me with up to {num_queries} search queries that I can input into a search engine to find info about this topic? Please do not qualify your response... Simply provide one search query per line and nothing else.",
        },
    ]

    response = chatbot.resend()
    lines = response.strip().split("\n")

    if len(lines) > num_queries:
        lines = lines[:num_queries]

    return lines


def uri_to_local(uri: str) -> str:
    """
    Convert a URI to a local file name. In this case, we typically will use an MD5 sum.

    Args:
        uri (str): The URI to convert.

    Returns:
        str: The MD5 sum of the URI.
    """
    uri_md5 = hashlib.md5(uri.encode("utf-8")).hexdigest()
    return uri_md5


class KnowledgeBaseFileCache:

    def __init__(self, folder_path: str, cache_file: str = "cache.json") -> None:
        """
        The KnowledgeBaseFileCache is a simple file-based cache for web content and local files. The cache stores the original HTML, PDF, or TXT content and tracks when (if ever) an agent actually accessed the content.

        Args:
            folder_path (str): The folder where the cache will be stored.
            cache_file (str, optional): The name of the cache file. Defaults to "cache.json".
        """
        self.root_path = folder_path
        self.root_parsed = os.path.join(folder_path, "parsed")
        self.root_original = os.path.join(folder_path, "original")
        self.cache_file = os.path.join(folder_path, cache_file)
        self.cache = self.load_cache()

    def save_state(self) -> None:
        """
        Saves the in-memory changes to the knowledge base to the JSON cache file.
        """
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f, cls=DjangoJSONEncoder)

    def load_cache(self) -> None:
        """
        Loads the cache from the cache file, or creates the relevant files and folders if one does not exist.
        """

        if not os.path.exists(self.root_path):
            os.makedirs(self.root_path)

        if not os.path.exists(self.root_parsed):
            os.makedirs(self.root_parsed)

        if not os.path.exists(self.root_original):
            os.makedirs(self.root_original)

        if not os.path.exists(self.cache_file):
            with open(self.cache_file, "w") as f:
                f.write("{}")

        with open(self.cache_file, "r") as f:
            return json.load(f)

    def in_cache(self, uri: str) -> bool:
        """
        Checks if a URI is in the cache already.

        Args:
            uri (str): The URI to check.

        Returns:
            bool: True if the URI is in the cache, False otherwise.
        """
        if uri in self.cache:
            return True
        return False

    def update_cache(
        self, uri: str, obtained_on: datetime, last_accessed: datetime
    ) -> None:
        """
        Updates the cache file for a given URI, specifically when it was obtained and last accessed.

        Args:
            uri (str): The URI to update.
            obtained_on (datetime): The date and time when the content was obtained.
            last_accessed (datetime): The date and time when the content was last accessed.
        """
        uri_md5 = uri_to_local(uri)
        self.cache[uri] = {
            "obtained_on": obtained_on,
            "last_accessed": last_accessed,
            "accessed": 0,
            "uri_md5": uri_md5,
        }
        self.save_state()

    def log_access(self, uri: str) -> None:
        """
        Saves the last accessed time and updates the accessed tracker for a given URI.

        Args:
            uri (str): The URI to update.
        """
        self.cache[uri]["last_accessed"] = datetime.now()
        self.cache[uri]["accessed"] = 1
        self.save_state()

    def get_unaccessed_content(self) -> list[str]:
        """
        Returns a list of URIs that have not been accessed by the agent.

        Returns:
            list[str]: A list of URIs that have not been accessed by the agent.
        """
        unaccessed = []
        for uri in self.cache:
            if self.cache[uri]["accessed"] == 0:
                unaccessed.append(uri)
        return unaccessed

    def get(self, uri: str) -> str:
        """
        Returns the content for a given URI. If the content is not in the cache, it will be scraped and added to the cache.

        Args:
            uri (str): The URI to get the content for.

        Returns:
            str: The content for the given URI.
        """
        uri_md5 = uri_to_local(uri)
        if uri in self.cache:
            with open(os.path.join(self.root_parsed, uri_md5), "r") as f:
                return f.read()
        else:
            scraper = WebpageAgent()

            content_raw = scraper.scrape(uri, text_only=False, body_only=False)
            with open(os.path.join(self.root_original, uri_md5), "w") as f:
                f.write(content_raw)

            content_parsed = scraper.scrape(uri, text_only=True, body_only=True)
            with open(os.path.join(self.root_parsed, uri_md5), "w") as f:
                f.write(content_parsed)

            self.update_cache(uri, datetime.now(), datetime.now())

            return content_parsed

    def add_content(self, content: str, uri: str = None) -> None:
        """
        Adds content to cache.

        Args:
            content (str): The content to add to the cache.
            uri (str, optional): The URI to use for the content. Defaults to None, in which case an MD5 sum of the content will be used.
        """
        if uri is None:
            uri = hashlib.md5(content.encode("utf-8")).hexdigest()
        uri_md5 = uri_to_local(uri)
        with open(os.path.join(self.root_parsed, uri_md5), "w") as f:
            f.write(content)
        self.update_cache(uri, datetime.now(), datetime.now())

    def add_content_from_file(self, filepath: str, uri: str = None) -> None:
        """
        Adds content from a text file to the cache.

        Args:
            filepath (str): The path to the file to add to the cache.
            uri (str, optional): The URI to use for the content. Defaults to None, in which case an MD5 sum of the content will be used.
        """
        with open(filepath, "r") as f:
            content = f.read()
        self.add_content(content, uri)
