"""
This is an experimental approach to tracking information regardless of source type. It will also power more than recent updates. Here's how it works...

1. All "Content Sources" (a new class type that obtains content) will send content directly to the Facts DB.
2. The "Facts DB" will then extract all relevant facts for a prediction or research theme. It will keep cache the original content, will track the sources, and will also input all the facts into a RAG database.
3. We can then query the DB for relevant facts on an ad hoc basis, rather than only for new content.

"""

import os
import json
import hashlib
import re

# Using JSONEncoder to be consistent with the Emerging Trajectories website and platform.
from django.core.serializers.json import DjangoJSONEncoder

from phasellm.llms import OpenAIGPTWrapper, ChatBot, ChatPrompt
from phasellm.agents import WebpageAgent, WebSearchAgent

from datetime import datetime, timedelta

from . import Client
from .crawlers import crawlerPlaywright
from .prompts import *
from .news import NewsAPIAgent, RSSAgent, FinancialTimesAgent

import chromadb
import chromadb.utils.embedding_functions as embedding_functions

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

fact_system_prompt = """You are a researcher helping extract facts about {topic}, trends, and related observations. We will give you a piece of content scraped on the web. Please extract facts from this. Each fact should stand on its own, and can be several sentences long if need be. You can have as many facts as needed. For each fact, please start it as a new line with "---" as the bullet point. For example:

--- Fact 1... This is the fact.
--- Here is a second fact.
--- And a third fact.

Please do not include new lines between bullet points. Make sure you write your facts in ENGLISH. Translate any foreign language content/facts/observations into ENGLISH.

We will simply provide you with content and you will just provide facts."""


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


# TODO If this works, it should be an agent with setllm() supported, etc.
# TODO Right now, we don't actually save sources. It's an important feature (track reliability, etc. too!) but we want to ensure the POC works well first.
class FactRAGFileCache:

    def __init__(
        self,
        folder_path: str,
        openai_api_key: str,
        cache_file: str = "cache.json",
        rag_db_folder="cdb",
        crawler=None,
    ) -> None:
        """
        This is a RAG-based fact database. We build a database of facts available in JSON and via RAG and use this as a basic search engine for information. We use ChromaDB to index all facts, but also maintain a list of facts, sources, etc. in a JSON file. Finally, we keep a cache of all content and assume URLs do not get updated; we'll change this process in the future.

        Args:
            folder_path (str): The folder where everything will be stored.
            openai_api_key (str): The OpenAI API key. Used for RAG embeddings.
            cache_file (str, optional): The name of the cache file. Defaults to "cache.json".
            rag_db_folder (str, optional): The folder where the ChromaDB database will be stored. Defaults to "cdb".
            crawler (optional): The crawler to use. Defaults to None, in which case a Playwright crawler will be used.
        """
        self.root_path = folder_path
        self.root_parsed = os.path.join(folder_path, "parsed")
        self.root_original = os.path.join(folder_path, "original")
        self.cache_file = os.path.join(folder_path, cache_file)
        self.rag_db_folder = os.path.join(folder_path, rag_db_folder)
        self.openai_api_key = openai_api_key

        # Use the same default crawler for all other agents.
        # TODO Eventually we'll want to have an array agents we use to get content.
        if crawler is None:
            self.crawler = crawlerPlaywright()
        else:
            self.crawler = crawler

        # Set up / load Chroma DB
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=self.openai_api_key, model_name="text-embedding-3-small"
        )
        self.chromadb_client = chromadb.PersistentClient(path=self.rag_db_folder)
        self.facts_rag_collection = self.chromadb_client.get_or_create_collection(
            name="facts", embedding_function=openai_ef
        )

        # Set up / load cache
        self.cache = self.load_cache()

    def get_facts_as_dict(self, n_results=-1, min_date: datetime = None) -> list:
        """
        Get all facts as a list.

        Args:
            n_results (int, optional): The number of results to return. Defaults to -1, in which case all results are returned.

        Returns:
            list: A list of fact dictionaries containing content, source, and added (the date string for when the fact was added).
        """

        if n_results == -1:
            all_facts_raw = self.facts_rag_collection.peek(
                limit=self.facts_rag_collection.count()
            )
        else:
            all_facts_raw = self.facts_rag_collection.peek(n_results)

        facts = {}
        for i in range(0, len(all_facts_raw["documents"])):
            fact_id = all_facts_raw["ids"][i]
            fact_content = all_facts_raw["documents"][i]
            fact_source = all_facts_raw["metadatas"][i]["source"]
            fact_datetime = all_facts_raw["metadatas"][i]["datetime_string"]
            fact_timestamp = all_facts_raw["metadatas"][i]["added_on_timestamp"]

            add_fact = False
            if min_date is None:
                add_fact = True
            else:
                if min_date.timestamp() <= fact_timestamp:
                    add_fact = True

            if add_fact:

                facts[fact_id] = {
                    "content": fact_content,
                    "source": fact_source,
                    "added": fact_datetime,
                    "added_on_timestamp": fact_timestamp,
                }

        return facts

    def get_facts_as_list(self) -> list:
        """
        Get all facts as a list.

        Args:
            None

        Returns:
            list: A list of facts (as strings).
        """

        all_facts_raw = self.facts_rag_collection.peek(
            self.facts_rag_collection.count()
        )

        facts = []
        for d in all_facts_raw["documents"]:
            facts.append(d)

        return facts

    def count_facts(self) -> int:
        """
        Returns the number of facts in the knowledge database.

        Returns:
            int: The number of facts in the knowledge database.
        """
        return self.facts_rag_collection.count()

    def get_fact_details(self, fact_id: str) -> dict:
        """
        Returns similar structure as query_to_fact_list() but for a specific fact ID. Returns NONE otherwise.

        Args:
            fact_id (str): The fact ID to get.

        Returns:
            dict: A dictionary with the content, source, added date, and added timestamp.
        """

        results = self.facts_rag_collection.get(fact_id)
        if len(results["ids"]) == 0:
            return None

        fact_content = results["documents"][0]
        fact_source = results["metadatas"][0]["source"]
        fact_datetime = results["metadatas"][0]["datetime_string"]
        fact_timestamp = results["metadatas"][0]["added_on_timestamp"]

        return {
            "content": fact_content,
            "source": fact_source,
            "added": fact_datetime,
            "added_on_timestamp": fact_timestamp,
        }

    def query_to_fact_list(
        self, query: str, n_results: int = 10, since_date: datetime = None
    ) -> dict:
        """
        Takes a query and finds the closest semantic matches to the query in the knowledge base.

        Args:
            query (str): The query to search for.
            n_results (int, optional): The number of results to return. Defaults to 10.
            since_date (datetime, optional): The date to search from. Defaults to None, in which case all dates are searched.

        Returns:
            dict: A list of the facts found, with the key being the fact ID and each fact having its source, add date, and content info.
        """

        r = []

        if n_results == -1:
            n_results = self.count_facts()

        if since_date is None:
            r = self.facts_rag_collection.query(
                query_texts=[query], n_results=n_results
            )
        else:
            r = self.facts_rag_collection.query(
                query_texts=[query],
                n_results=n_results,
                where={"added_on_timestamp": {"$gt": since_date.timestamp()}},
            )

        facts = {}

        print(r["metadatas"][0])

        for i in range(0, len(r["ids"][0])):
            fact_id = r["ids"][0][i]
            fact_content = r["documents"][0][i]
            fact_source = r["metadatas"][0][i]["source"]
            fact_datetime = r["metadatas"][0][i]["datetime_string"]
            fact_timestamp = (r["metadatas"][0][i]["added_on_timestamp"],)

            facts[fact_id] = {
                "content": fact_content,
                "source": fact_source,
                "added": fact_datetime,
                "added_on_timestamp": fact_timestamp,
            }

        return facts

    def query_to_fact_content(
        self, query: str, n_results: int = 10, since_date=None, skip_separator=False
    ) -> str:
        """
        Takes a query and finds the closest semantic matches to the query in the knowledge base.

        Args:
            query (str): The query to search for.
            n_results (int, optional): The number of results to return. Defaults to 10.
            since_date ([type], optional): The date to search from. Defaults to None, in which case all dates are searched.
            skip_separator (bool, optional): Whether to prepend and append a note horizontal line and title to the string being returned. Defaults to False.

        Returns:
            str: The content of the facts found, along with the fact IDs.

        """

        facts = self.query_to_fact_list(query, n_results, since_date)

        if len(facts) == 0:
            return ""

        fact_content = ""
        if not skip_separator:
            fact_content = """--- START FACTS ---------------------------\n"""

        for key, fact in facts.items():
            fact_content += key + ": " + fact["content"] + "\n"

        if not skip_separator:
            fact_content += """--- END FACTS ---------------------------\n"""

        return fact_content

    def get_all_recent_facts(self, days: float = 1, skip_separator=False) -> str:
        """
        Returns a list of all facts and sources added in the last n days.

        Args:
            days (float, optional): The number of days to search back. Defaults to 1. Can be fractional as well.
            skip_separator (bool, optional): Whether to prepend and append a note horizontal line and title to the string being returned. Defaults to False.

        Returns:
            str: The content of the facts found, along with the fact IDs.
        """

        fact_content = ""

        if not skip_separator:
            fact_content = """--- START FACTS ---------------------------\n"""

        # min_date_timestamp = (datetime.now() - timedelta(days=days)).timestamp()
        # applicable_facts = self.query_to_fact_list("", -1, min_date_timestamp)

        min_date = datetime.now() - timedelta(days=days)
        min_date_timestamp = min_date.timestamp()

        # This "" approach doesn't work because OpenAI errors out.
        # applicable_facts = self.query_to_fact_list("", -1, min_date)
        applicable_facts = self.get_facts_as_dict(-1, min_date)

        for key, fact in applicable_facts.items():
            if fact["added_on_timestamp"] > min_date_timestamp:
                fact_content += key + ": " + fact["content"] + "\n"

        if not skip_separator:
            fact_content += """--- END FACTS ---------------------------\n"""

        return fact_content

    def get_fact_source(self, fact_id: str) -> str:
        """
        Returns the source of a fact given its ID.

        Args:
            fact_id: The fact ID to get.

        Returns:
            str: The source of the fact.
        """

        results = self.facts_rag_collection.get(fact_id)
        if len(results["ids"]) == 0:
            new_id = fact_id.lower()
            results = self.facts_rag_collection.get(new_id)

        if len(results["ids"]) == 0:
            raise ValueError(f"Fact ID {fact_id} not found in the knowledge database.")

        if "source" in results["metadatas"][0]:
            return results["metadatas"][0]["source"]

        raise ValueError(
            f"Fact ID {fact_id} does not have a source in the knowledge database."
        )

    def get_fact_content(self, fact_id: str) -> str:
        """
        Returns the content of a fact given its ID.

        Args:
            fact_id: The fact ID to get.

        Returns:
            str: The content of the fact.
        """

        results = self.facts_rag_collection.get(fact_id)
        if len(results["ids"]) == 0:
            new_id = fact_id.lower()
            results = self.facts_rag_collection.get(new_id)

        if len(results["ids"]) == 0:
            raise ValueError(f"Fact ID {fact_id} not found in the knowledge database.")

        return results["documents"][0]

    def add_fact(self, fact: str, url: str) -> bool:
        """
        Adds a fact to the knowledge base.

        Args:
            fact (str): The fact to add.
            url (str): The URL source of the fact.

        Returns:
            bool: True if the fact was added, False otherwise.
        """
        return self.add_fact([fact], [url])

    def add_facts(self, facts: list, sources: list) -> bool:
        """
        Adds a facts to the knowledge base.

        Args:
            facts (list): List of strings. Each string is a fact.
            sources (list): List of sources (e.g., URLs) for the facts.

        Returns:
            bool: True if the facts were added, False otherwise.
        """

        fact_id_start = self.facts_rag_collection.count() + 1

        added_now = datetime.now()
        added_now_timestamp = added_now.timestamp()
        added_now_string = added_now.strftime("%Y-%m-%d %H:%M:%S")

        fact_ids = []
        metadatas = []
        for i in range(0, len(facts)):
            fact_ids.append(f"f{fact_id_start + i}")
            metadatas.append(
                {
                    "added_on_timestamp": added_now_timestamp,
                    "datetime_string": added_now_string,
                    "source": sources[i],
                }
            )

        self.facts_rag_collection.add(
            documents=facts, ids=fact_ids, metadatas=metadatas
        )

        return True

    def facts_from_url(self, url: str, topic: str) -> None:
        """
        Given a URL, extract facts from it and save them to ChromaDB and the facts dictionary. Also returns the facts in an array, in case one wants to analyze new facts.

        Args:
            url (str): Location of the content.
            topic (str): a brief description of the research you are undertaking.
        """

        content = self.get(url)

        llm = OpenAIGPTWrapper(self.openai_api_key, model="gpt-4-turbo-preview")
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
        sources = []

        for line in lines:
            if line[0:4] == "--- ":
                fact = line[4:]
                facts.append(fact)
                sources.append(url)

        if len(facts):
            return self.add_facts(facts, sources)
        return False

    # This builds facts based on RSS feeds.
    def new_get_rss_links(self, rss_url, topic) -> None:
        """
        Crawls an RSS feed and its posts.

        Args:
            rss_url (str): The URL of the RSS feed.
            topic (str): a brief description of the research you are undertaking.
        """

        rss_agent = RSSAgent(rss_url, crawler=self.crawler)
        urls = rss_agent.get_news_as_list()

        for url in urls:
            if not self.in_cache(url):
                print("RSS RESULT: " + url)
                try:
                    self.facts_from_url(url, topic)
                except:
                    print("Error; failed to get content from " + url)

    # This builds facts based on news articles.
    def new_get_new_info_news(
        self,
        newsapi_api_key,
        topic,
        queries,
        top_headlines=False,
    ) -> None:
        """
        Uses the News API to find new information and extract facts from it.

        Args:
            newsapi_api_key (str): The News API key.
            topic (str): a brief description of the research you are undertaking.
            queries (list[str]): A list of queries to search for.
            top_headlines (bool, optional): Whether to search for top headlines. Defaults to False.
        """

        news_agent = NewsAPIAgent(
            newsapi_api_key, top_headlines=top_headlines, crawler=self.crawler
        )

        for q in queries:
            results = news_agent.get_news_as_list(q)
            for result in results["articles"]:
                url = result["url"]
                if not self.in_cache(url):
                    print("NEWS RESULT: " + url)
                    self.facts_from_url(url, topic)

    # POC for FT
    def get_ft_news(self, ft_user, ft_pass, topic) -> None:
        """
        Uses the Financial Times Agent to find new information and extract facts from it.

        Args:
            ft_user (str): The Financial Times username.
            ft_pass (str): The Financial Times password.
            topic (str): a brief description of the research you are undertaking.
        """

        fta = FinancialTimesAgent(ft_user, ft_pass)
        urls, html_content, text_content = fta.get_news()

        if len(urls) != len(text_content):
            raise ValueError("URLs and text content are not the same length.")

        for i in range(0, len(urls)):
            url = urls[i]
            content = text_content[i]

            if not self.in_cache(url):
                print("FT RESULT: " + url)
                self.force_content(url, content)
                self.facts_from_url(url, topic)

    # This builds facts based on all the google searches.
    def new_get_new_info_google(
        self,
        google_api_key,
        google_search_id,
        google_search_queries,
        topic,
    ) -> None:
        """
        Uses Google search to find new information and extract facts from it.

        Args:
            google_api_key (str): The Google API key.
            google_search_id (str): The Google search ID.
            google_search_queries (list[str]): A list of queries to search for.
            topic (str): a brief description of the research you are undertaking.
        """

        self.google_api_key = google_api_key
        self.google_search_id = google_search_id
        self.google_search_queries = google_search_queries

        webagent = WebSearchAgent(api_key=self.google_api_key)

        for google_search_query in self.google_search_queries:

            results = webagent.search_google(
                query=google_search_query,
                custom_search_engine_id=self.google_search_id,
                num=_DEFAULT_NUM_SEARCH_RESULTS,
            )

            for result in results:
                if not self.in_cache(result.url):
                    try:
                        print("SEARCH RESULT: " + result.url)
                        # page_content = self.get(result.url)
                        self.facts_from_url(result.url, topic)
                        # print(page_content)
                    except Exception as e:
                        print(f"Failed to get content from {result.url}\n{e}")

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

    def force_content(self, uri: str, content: str, check_exists: bool = True) -> bool:
        """
        Forces a specific URI to have specific content (both HTML and text content). Used to fill old links that we don't actually want to crawl.

        Args:
            uri (str): The URI to force content for.
            content (str): The content to force.
            check_exists (bool): checks if content has already been included in the cache before forcing the new content.

        Returns:
            bool: True if the content was forced, False otherwise.
        """

        # If the content already exists and we avoid overwrites, then we don't want to overwrite it.
        if check_exists and self.in_cache(uri):
            return False

        uri_md5 = uri_to_local(uri)
        with open(os.path.join(self.root_original, uri_md5), "w") as f:
            f.write(content)
        with open(os.path.join(self.root_parsed, uri_md5), "w") as f:
            f.write(content)

        self.update_cache(uri, datetime.now(), datetime.now())
        self.log_access(uri)

        return True

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

            try:
                content, text = self.crawler.get_content(uri)
            except Exception as e:
                print(f"Failed to get content from {uri}\n{e}")
                content = ""
                text = ""

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


class FactBot:

    def __init__(
        self,
        knowledge_db: FactRAGFileCache,
        openai_api_key: str = None,
        chatbot: ChatBot = None,
    ) -> None:
        """
        The FactBot is like a ChatBot but enables you to ask questions that reference an underlying RAG database (KnowledgeBaseFileCache), which then enables the chatbot to cite sourcable facts.

        Args:
            knowledge_db (FactRAGFileCache): The knowledge database to use.
            openai_api_key (str, optional): The OpenAI API key. Defaults to None.
            chatbot (ChatBot, optional): The PhaseLLB chatbot to use. Defaults to None, in which case an OpenAI chatbot is used (and the OpenAI API key must be provided).
        """
        if openai_api_key is None and chatbot is None:
            raise ValueError("One of openai_api_key or chatbot must be provided.")

        if chatbot is not None:
            self.chatbot = chatbot
        else:
            llm = OpenAIGPTWrapper(openai_api_key, model="gpt-4-turbo-preview")
            self.chatbot = ChatBot(llm)
            self.chatbot.messages = [
                {"role": "system", "content": system_prompt_question_continuous}
            ]

        self.knowledge_db = knowledge_db

    def ask(self, question: str, clean_sources: bool = True) -> str:
        """
        Ask a question to the FactBot. This will query the underlying knowledge database and use the returned facts to answer the question.

        Args:
            question (str): The question to ask.
            clean_sources (bool, optional): Whether to clean the sources in the response. Defaults to True; in this case, it will replace fact IDs with relevant source links at the end of the response.

        Returns:
            str: The response to the question.
        """
        message = self.knowledge_db.query_to_fact_content(question) + "\n\n" + question
        response = self.chatbot.chat(message)
        if clean_sources:
            return clean_fact_citations(self.knowledge_db, response)
        else:
            return response

    def source(self, fact_id: str) -> str:
        """
        Returns the URL source for a given fact ID.

        Args:
            fact_id (str): The fact ID to get the source for.

        Returns:
            str: The URL source for the given fact ID.
        """
        return self.knowledge_db.get_fact_source(fact_id)

    def clean_and_source_to_html(
        self, text_to_clean: str, start_count: int = 0
    ) -> list:
        """
        Returns a formatted response with sourced HTML. This is used for emergingtrajectories.com and acts as a base for anyone else wanting to build similar features.

        Args:
            text_to_clean: The text to clean/cite/source.
            start_count: The starting count for the sources.

        Returns:
            list: two strings -- the actual response in the first case, and the sources in the second case, and an integer representing the new source count.
        """

        pattern = r"\[f[\d\s\,f]+\]"
        new_text = ""
        sources_text = ""
        ref_ctr = start_count
        last_index = 0

        for match in re.finditer(pattern, text_to_clean, flags=re.IGNORECASE):

            if match.group(0).find(",") == -1:
                ref_ctr += 1
                ref = match.group(0)[1:-1].strip()
                ref = ref.lower()

                new_text += text_to_clean[last_index : match.start()]
                new_text += f"""<a class='source_link' target='_blank' href='{self.source(ref)}'>{ref_ctr}</a>"""

                # Save the source
                fact_text = self.knowledge_db.get_fact_content(ref)
                new_source_text = f"""<span class='fact_span'><b>{ref_ctr}:</b> {fact_text} <a href='{self.source(ref)}' target='_blank'>View Source</a></span>"""
                sources_text += new_source_text + "\n"

                last_index = match.end()
            else:
                refs = match.group(0)[1:-1].split(",")
                ref_arr = []
                ref_str = ""
                for ref in refs:
                    ref = ref.strip()
                    ref = ref.lower()
                    ref_ctr += 1
                    ref_arr.append(str(ref_ctr))

                    # Add the source to the text
                    new_text_source_num = f"""<a class='source_link' target='_blank' href='{self.source(ref)}'>{ref_ctr}</a>"""
                    ref_str += " " + new_text_source_num

                    # Save the source
                    fact_text = self.knowledge_db.get_fact_conteont(ref)
                    new_source_text = f"""<span class='fact_span'><b>{ref_ctr}:</b> ${fact_text} <a href='{self.source(ref)}' target='_blank'>View Source</a></span>"""
                    sources_text += new_source_text + "\n"

                new_text += text_to_clean[last_index : match.start()] + ref_str
                last_index = match.end()

        new_text += text_to_clean[last_index:]

        return new_text, sources_text, ref_ctr


def clean_fact_citations(knowledge_db: FactRAGFileCache, text_to_clean: str) -> str:
    """
    Converts fact IDs referenced in a piece of text to relevant source links, appending sources as end notes in the document/text.

    Args:
        knowledge_db (FactRAGFileCache): The knowledge database to use for fact lookups.
        text_to_clean (str): The text to clean.

    Returns:
        str: The cleaned text.
    """
    bot = FactBot(knowledge_db, knowledge_db.openai_api_key)
    pattern = r"\[f[\d\s\,f]+\]"
    new_text = ""
    ref_ctr = 0
    last_index = 0
    sources_list = ""
    for match in re.finditer(pattern, text_to_clean):
        if match.group(0).find(",") == -1:
            ref_ctr += 1
            ref = match.group(0)[1:-1].strip()
            new_text += text_to_clean[last_index : match.start()]
            new_text += f"[{ref_ctr}]"
            sources_list += f"{ref_ctr} :: " + bot.source(f"{ref}") + "\n"
            last_index = match.end()
        else:
            refs = match.group(0)[1:-1].split(",")
            ref_arr = []
            for ref in refs:
                ref = ref.strip()
                ref_ctr += 1
                ref_arr.append(str(ref_ctr))
                sources_list += f"{ref_ctr} :: " + bot.source(f"{ref}") + "\n"
            ref_str = "[" + ", ".join(ref_arr) + "]"
            new_text += text_to_clean[last_index : match.start()] + ref_str
            last_index = match.end()

    new_text += text_to_clean[last_index:]

    if ref_ctr == 0:
        return text_to_clean
    else:
        return new_text + "\n\nSources:\n" + sources_list
