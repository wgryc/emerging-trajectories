"""
Facts agent. Similar to knowledge agent but simply provides a list of facts and associated sources.

This abstracts away the fact generation from forecast creation, thus allowing us to test different prompting strategies and LLMs.
"""

import os
import json
import hashlib
import re

# Using JSONEncoder to be consistent with the Emerging Trajectories website and platform.
from django.core.serializers.json import DjangoJSONEncoder

from phasellm.llms import OpenAIGPTWrapper, ChatBot, ChatPrompt
from phasellm.agents import WebpageAgent, WebSearchAgent

from datetime import datetime

from . import Client
from .crawlers import crawlerPlaywright
from phasellm.llms import OpenAIGPTWrapper, ChatBot

# Number of search results to return from web searche (default value).
_DEFAULT_NUM_SEARCH_RESULTS = 10

facts_base_system_prompt = """You are a researcher tasked with helping forecast economic and social trends. The title of our research project is: {statement_title}.

The project description is as follows...
{statement_description}

We will provide you with content from reports and web pages that is meant to help with the above. We will ask you to review these documents, create a set of bullet points to inform your thinking. Rather than using bullet points, please list each as F1, F2, F3, etc... So that we can reference it.

The content we provided you contains source numbers in the format 'SOURCE: #'. When you extract facts, please include the citation in square brackets, with the #, like [#], but replace "#" with the actual Source # from the crawled content we are providing you.

For example, if you are referring to a fact that came under --- SOURCE: 3 ---, you would write something like: "Data is already trending to hotter temperatures [3]." Do not include the "#" in the brackets, just the number.

Thus, a bullet point would look like this:
F1: (information) [1]
F2: (information) [1]
F3: (information) [2]

... and so on, where F1, F2, F3, etc. are facts, and [1], [2] are the source documents you are extracting the facts from.
"""

facts_base_user_prompt = """Today's date is {the_date}. We will now provide you with all the content we've managed to collect. 

----------------------
{scraped_content}
----------------------

Please think step-by-step by (a) extracting critical bullet points from the above, and (b) share any insights you might have based on the facts.

The content we provided you contains source numbers in the format 'SOURCE: #'. When you extract facts, please include the citation in square brackets, with the #, like [#], but replace "#" with the actual Source # from the crawled content we are providing you.

For example, if you are referring to a fact that came under --- SOURCE: 3 ---, you would write something like: "Data is already trending to hotter temperatures [3]." Do not include the "#" in the brackets, just the actual number.

DO NOT PROVIDE A FORECAST, BUT SIMPLY STATE AND SHARE THE FACTS AND INSIGHTS YOU HAVE GATHERED.
"""


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


# TODO If this works, it should be an agent with setllm() supported, etc.
class FactBaseFileCache:

    def __init__(
        self, folder_path: str, cache_file: str = "cache.json", crawler=None
    ) -> None:
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

        if crawler is None:
            self.crawler = crawlerPlaywright()
        else:
            self.crawler = crawler

    # TODO: this function is a new one compared to the KnowledgeBaseFileCache
    # TODO: refactor this + code where we run one query
    def summarize_new_info_multiple_queries(
        self,
        statement,
        chatbot,
        google_api_key,
        google_search_id,
        google_search_queries,
        fileout=None,
    ) -> str:

        self.google_api_key = google_api_key
        self.google_search_id = google_search_id
        self.google_search_queries = google_search_queries

        webagent = WebSearchAgent(api_key=self.google_api_key)

        scraped_content = ""
        added_new_content = False

        # We store the accessed resources and log access only when we successfully submit a forecast. If anything fails, we'll review those resources again during the next forecasting attempt.
        accessed_resources = []

        ctr = 0
        ctr_to_source = {}

        for google_search_query in self.google_search_queries:

            results = webagent.search_google(
                query=google_search_query,
                custom_search_engine_id=self.google_search_id,
                num=_DEFAULT_NUM_SEARCH_RESULTS,
            )

            added_new_content = False

            for result in results:
                if not self.in_cache(result.url):
                    ctr += 1
                    added_new_content = True

                    try:
                        page_content = self.get(result.url)
                        print(page_content)
                    except Exception as e:
                        print(f"Failed to get content from {result.url}\n{e}")
                        self.force_empty(result.url)
                        page_content = ""

                    accessed_resources.append(result.url)
                    # knowledge_base.log_access(result.url)

                    scraped_content += (
                        f"{page_content}\n\n--- SOURCE: {ctr}-------------------\n\n"
                    )
                    ctr_to_source[ctr] = result.url

        # We also check the knowledge base for content that was added manually.
        unaccessed_uris = self.get_unaccessed_content()
        for ua in unaccessed_uris:
            added_new_content = True
            ctr += 1
            page_content = self.get(ua)

            accessed_resources.append(ua)
            # knowledge_base.log_access(ua)

            scraped_content += (
                f"{page_content}\n\n--- SOURCE: {ctr}-------------------\n\n"
            )
            ctr_to_source[ctr] = ua

        if not added_new_content:
            print("No new content added to the forecast.")
            return None

        the_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        prompt_template = ChatPrompt(
            [
                {"role": "system", "content": facts_base_system_prompt},
                {"role": "user", "content": facts_base_user_prompt},
            ]
        )

        chatbot.messages = prompt_template.fill(
            statement_title=statement.title,
            statement_description=statement.description,
            statement_fill_in_the_blank=statement.fill_in_the_blank,
            scraped_content=scraped_content,
            the_date=the_date,
        )

        assistant_analysis = chatbot.resend()
        assistant_analysis_sourced = clean_citations(assistant_analysis, ctr_to_source)

        print("\n\n\n")
        print(assistant_analysis_sourced)

        if fileout is not None:
            with open(fileout, "w") as w:
                w.write(assistant_analysis_sourced)

        for ar in accessed_resources:
            self.log_access(ar)

        return assistant_analysis_sourced

    # TODO: this function is a new one compared to the KnowledgeBaseFileCache
    def summarize_new_info(
        self,
        statement,
        chatbot,
        google_api_key,
        google_search_id,
        google_search_query,
        fileout=None,
    ) -> str:

        self.google_api_key = google_api_key
        self.google_search_id = google_search_id
        self.google_search_query = google_search_query

        webagent = WebSearchAgent(api_key=self.google_api_key)
        results = webagent.search_google(
            query=self.google_search_query,
            custom_search_engine_id=self.google_search_id,
            num=_DEFAULT_NUM_SEARCH_RESULTS,
        )

        scraped_content = ""

        added_new_content = False

        # We store the accessed resources and log access only when we successfully submit a forecast. If anything fails, we'll review those resources again during the next forecasting attempt.
        accessed_resources = []

        ctr = 0
        ctr_to_source = {}

        for result in results:
            if not self.in_cache(result.url):
                ctr += 1
                added_new_content = True

                try:
                    page_content = self.get(result.url)
                    print(page_content)
                except Exception as e:
                    print(f"Failed to get content from {result.url}\n{e}")
                    self.force_empty(result.url)
                    page_content = ""

                accessed_resources.append(result.url)
                # knowledge_base.log_access(result.url)

                scraped_content += (
                    f"{page_content}\n\n--- SOURCE: {ctr}-------------------\n\n"
                )
                ctr_to_source[ctr] = result.url

        # We also check the knowledge base for content that was added manually.
        unaccessed_uris = self.get_unaccessed_content()
        for ua in unaccessed_uris:
            added_new_content = True
            ctr += 1
            page_content = self.get(ua)

            accessed_resources.append(ua)
            # knowledge_base.log_access(ua)

            scraped_content += (
                f"{page_content}\n\n--- SOURCE: {ctr}-------------------\n\n"
            )
            ctr_to_source[ctr] = ua

        if not added_new_content:
            print("No new content added to the forecast.")
            return None

        the_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        prompt_template = ChatPrompt(
            [
                {"role": "system", "content": facts_base_system_prompt},
                {"role": "user", "content": facts_base_user_prompt},
            ]
        )

        chatbot.messages = prompt_template.fill(
            statement_title=statement.title,
            statement_description=statement.description,
            statement_fill_in_the_blank=statement.fill_in_the_blank,
            scraped_content=scraped_content,
            the_date=the_date,
        )

        assistant_analysis = chatbot.resend()
        assistant_analysis_sourced = clean_citations(assistant_analysis, ctr_to_source)

        print("\n\n\n")
        print(assistant_analysis_sourced)

        if fileout is not None:
            with open(fileout, "w") as w:
                w.write(assistant_analysis_sourced)

        for ar in accessed_resources:
            self.log_access(ar)

        return assistant_analysis_sourced

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

    def force_empty(self, uri: str) -> None:
        """
        Saves an empty file for a given URI. Used when the page is erroring out.

        Args:
            uri (str): The URI to empty the cache for.
        """
        uri_md5 = uri_to_local(uri)

        with open(os.path.join(self.root_original, uri_md5), "w") as f:
            f.write("")
        with open(os.path.join(self.root_parsed, uri_md5), "w") as f:
            f.write("")

        self.update_cache(uri, datetime.now(), datetime.now())

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
            # scraper = WebpageAgent()

            # content_raw = scraper.scrape(uri, text_only=False, body_only=False)
            # with open(os.path.join(self.root_original, uri_md5), "w") as f:
            #    f.write(content_raw)

            # content_parsed = scraper.scrape(uri, text_only=True, body_only=True)
            # with open(os.path.join(self.root_parsed, uri_md5), "w") as f:
            #    f.write(content_parsed)

            content, text = self.crawler.get_content(uri)
            with open(os.path.join(self.root_original, uri_md5), "w") as f:
                f.write(content)
            with open(os.path.join(self.root_parsed, uri_md5), "w") as f:
                f.write(text)

            self.update_cache(uri, datetime.now(), datetime.now())

            return text

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
