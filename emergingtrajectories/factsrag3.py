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

# New libraries for FAISS, etc.
import numpy as np
import faiss
import pickle
import warnings

from openai import OpenAI
import tiktoken

# Using JSONEncoder to be consistent with the Emerging Trajectories website and platform.
from django.core.serializers.json import DjangoJSONEncoder

from phasellm.llms import OpenAIGPTWrapper, ChatBot, ChatPrompt
from phasellm.agents import WebpageAgent, WebSearchAgent

from datetime import datetime, timedelta

from . import Client
from .crawlers import crawlerPlaywright
from .prompts import *
from .news import NewsAPIAgent, RSSAgent, FinancialTimesAgent, NewsBingAgent
from .chunkers import *

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


class VectorDBDict:
    """
    This is a Python dictionary that gets converted into a FAISS index, and gets pickled to disk. This also comes with some "DB-like" features:
    (a) a unique, autoincrementing index
    (b) no ability to delete items (for now)
    (c) additional meta data you can save
    (d) TODO, maybe: locks to prevent concurrent writes

    Pickling presents some security issues, so please be careful if you are sharing pickled data files or being provided such files from others.
    """

    # Right now, we are only working with OpenAI's 'text-embedding-3-small' model.
    VECTOR_SIZE = 1536
    MAX_BATCH_SIZE = 100

    def __init__(
        self,
        db_file_path: str,
        openai_api_key: str,
        error_out_on_conflict: bool = False,
    ) -> None:
        """
        Initialize the database.

        Args:
            db_file_path (str): The path to the database file. Will be created if it does not exist.
            openai_api_key (str): The OpenAI API key.
            error_out_on_conflict (bool): If True, we will error out if the database tries to write to a file that doesn't align with the SHA hash of the DB file when it was loaded. Basically like a lock without actually being a lock. When it's set to False, it will simply print an error.
        """

        self.db_file_path = db_file_path
        self.error_out_on_conflict = error_out_on_conflict
        self.openai_api_key = openai_api_key
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        self.encoding = tiktoken.encoding_for_model("text-embedding-3-small")

        if not os.path.exists(db_file_path):
            dictionary_to_write = {
                "embeddings": faiss.IndexFlatL2(self.VECTOR_SIZE),
                "original_texts": [],
                "metadata": [],
            }
            pickle.dump(
                dictionary_to_write, open(db_file_path, "wb", pickle.HIGHEST_PROTOCOL)
            )

        self.db = pickle.load(open(db_file_path, "rb"))
        self.db_hash = self.get_file_sha256(db_file_path)

    def add_vectors(
        self, vectors: np.array, texts: list, metadata: list = None
    ) -> list:
        """
        Add vectors to the database.

        Args:
            vectors (np.array): The vectors to add.
            texts (list): The text for each vector.
            metadata (list, optional): The metadata for each vector.

        Returns:
            list: The IDs of the vectors.
        """

        start_index = len(self.db["original_texts"])

        if metadata is None:
            metadata = [{} for i in range(0, len(texts))]

        self.db["embeddings"].add(np.array(vectors))
        self.db["original_texts"] += texts
        self.db["metadata"] += metadata

        end_index = len(self.db["original_texts"])

        return list(range(start_index, end_index))

    def add_vector(self, vector: np.array, text: str, metadata: dict = None) -> int:
        """
        Adds a vector to the database.

        Args:
            vector (np.array): The vector to add.
            text (str): The text for the vector.
            metadata (dict): The metadata for the vector.

        Returns:
            int: The ID of the vector.
        """
        return self.add_vectors([vector], [text], [metadata])[0]

    def shorten_text(self, text, max_token_length: int = 8000) -> str:
        """
        Shortens text to a maximum token length. This is useful for OpenAI API calls, which have a token limit.

        Args:
            text (str): The text to shorten.
            max_token_length (int): The maximum token length.

        Returns:
            str: The shortened text; should ONLY be used for encoding.
        """

        new_text = text[:]
        while len(self.encoding.encode(new_text)) > max_token_length:
            if len(new_text) > 250:
                new_text = new_text[:-250] + "..."

        return new_text

    def add_texts(self, texts: list, metadata: list = None) -> list:
        """
        Adds text to the database. Calls an embedding function and then adds via add_vectors().

        Args:
            texts (list): The texts to add.
            metadata (list): Metadata to add.

        Returns:
            list: The IDs of the texts.
        """

        last_index = 0
        embeddings = []
        while last_index < len(texts):

            new_modifier = min(self.MAX_BATCH_SIZE, len(texts) - last_index)

            # In this approach, we keep the original text even though we encode the new text. We might want to revisit this to keep the new text.
            # TODO See above.
            text_subset = []
            text_subset_prechecked = texts[last_index : (last_index + new_modifier)]
            for t in text_subset_prechecked:
                t_new = self.shorten_text(t)
                text_subset.append(t_new)

            response = self.openai_client.embeddings.create(
                input=text_subset, model="text-embedding-3-small"
            )
            embeddings += list(map(lambda x: x.embedding, response.data))

            last_index += new_modifier

        return self.add_vectors(embeddings, texts, metadata)

    def add_text(self, text: str, metadata: dict = None) -> int:
        """
        Adds text to the database. Calls an embedding function and then adds via add_vectors().

        Args:
            text (str): The text to add.
            metadata (dict): Metadata to add.

        Returns:
            str: The ID of the text.
        """
        return self.add_texts([text], [metadata])[0]

    def get_file_sha256(self, file_path: str) -> str:
        """
        Get the SHA256 hash of a file. We use this to warn the user if/when the database is being saved and potetially conflicts with the underlying file. Written by GitHub copilot! 🙌

        Args:
            file_path (str): The path to the file.

        Returns:
            str: The SHA256 hash of the file.
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def save(self):
        """
        Save the database to disk.
        """
        file_sha = self.get_file_sha256(self.db_file_path)
        if file_sha != self.db_hash:
            if self.error_out_on_conflict:
                raise ValueError(
                    "The database file has been modified since it was loaded. Please reload the database and try again."
                )
            else:
                warnings.warn(
                    "WARNING: The database file has been modified since it was loaded. Please reload the database and try again."
                )

        pickle.dump(self.db, open(self.db_file_path, "wb", pickle.HIGHEST_PROTOCOL))
        self.db_hash = self.get_file_sha256(self.db_file_path)

    def count(self) -> int:
        """
        Returns the size of the DB.
        """
        return len(self.db["original_texts"])

    def get(self, index: int) -> dict:
        """
        Returns the text and metadata for a given index.

        Args:
            index (int): The index to get.

        Returns:
            dict: The text and metadata for the index.
        """

        text = self.db["original_texts"][index]
        metadata = self.db["metadata"][index]

        return text, metadata

    def query(self, text: str, n: int = 10) -> list:
        """
        Returns the closest vector IDs to a specific query/text.

        Args:
            text (str): The text to search for.
            n (int): The number of results to return.

        Returns:
            list: The IDs of the vectors (in order of closest to farthest).
        """

        q_response = self.openai_client.embeddings.create(
            input=[text], model="text-embedding-3-small"
        )
        q_embedding = q_response.data[0].embedding
        query_array = np.array([q_embedding])

        D, I = self.db["embeddings"].search(query_array, n)
        return list(I[0])

    def query_min_date(
        self, text: str, min_date: datetime, n: int = 10, date_field: str = "datetime"
    ) -> list:
        """
        Returns the closest vector IDs to a specific query/text, with a minimum date filter.

        Args:
            text (str): The text to search for.
            min_date (datetime): The minimum date to search from.
            n (int): The number of results to return.
            date_field: The field in the metadata to use for the date.

        Returns:
            list: The IDs of the vectors (in order of closest to farthest).
        """

        all_indices_sorted = self.query(text, self.count())

        return_me = []
        for i in range(0, len(all_indices_sorted)):
            if self.db["metadata"][all_indices_sorted[i]][date_field] >= min_date:
                return_me.append(all_indices_sorted[i])

            if len(return_me) >= n:
                break

        return return_me


# TODO If this works, it should be an agent with setllm() supported, etc.
# TODO Right now, we don't actually save sources. It's an important feature (track reliability, etc. too!) but we want to ensure the POC works well first.
class FactRAGFileCache:

    def __init__(
        self,
        folder_path: str,
        openai_api_key: str,
        cache_file: str = "cache.json",
        rag_db_file="vector_db.pickle",
        crawler=None,
        chunker=None,
    ) -> None:
        """
        This is a RAG-based fact database. We build a database of facts available in JSON and via RAG and use this as a basic search engine for information. We use our own DB to index all facts, but also maintain a list of facts, sources, etc. in a JSON file. Finally, we keep a cache of all content and assume URLs do not get updated; we'll change this process in the future.

        Args:
            folder_path (str): The folder where everything will be stored.
            openai_api_key (str): The OpenAI API key. Used for RAG embeddings.
            cache_file (str, optional): The name of the cache file. Defaults to "cache.json".
            rag_db_folder (str, optional): The folder where the database will be stored. Defaults to "cdb".
            crawler (optional): The crawler to use. Defaults to None, in which case a Playwright crawler will be used.
            chunker (optional): The sort of chunker to use. Defaults to None, in which case a GPT-4 chunker will be used.
        """
        self.root_path = folder_path
        self.root_parsed = os.path.join(folder_path, "parsed")
        self.root_original = os.path.join(folder_path, "original")
        self.cache_file = os.path.join(folder_path, cache_file)
        self.rag_db_file = os.path.join(folder_path, rag_db_file)
        self.openai_api_key = openai_api_key

        if chunker is None:
            self.chunker = ChunkerGPT4(openai_api_key)
        else:
            self.chunker = chunker

        # Use the same default crawler for all other agents.
        # TODO Eventually we'll want to have an array agents we use to get content.
        if crawler is None:
            self.crawler = crawlerPlaywright()
        else:
            self.crawler = crawler

        # Set up / load cache
        self.cache = self.load_cache()

        # Vector DB.
        self.vector_db = VectorDBDict(self.rag_db_file, self.openai_api_key)

    def get_facts_as_dict(self, n_results=-1, min_date: datetime = None) -> list:
        """
        Get all facts as a list.

        Args:
            n_results (int, optional): The number of results to return. Defaults to -1, in which case all results are returned.

        Returns:
            list: A list of fact dictionaries containing content, source, and added (the date string for when the fact was added).
        """

        all_facts_raw = self.vector_db.db["original_texts"]
        all_metadata_raw = self.vector_db.db["metadata"]

        facts = {}
        for i in range(0, len(all_facts_raw)):
            fact_id = f"f{i}"
            fact_content = all_facts_raw[i]
            fact_source = all_metadata_raw[i]["source"]
            fact_datetime_string = all_metadata_raw[i]["datetime"].strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            fact_timestamp = all_metadata_raw[i]["added_on_timestamp"]
            fact_datetime = all_metadata_raw[i]["datetime"]

            add_fact = False
            if min_date is None:
                add_fact = True
            else:
                if min_date <= fact_datetime:
                    add_fact = True

            if add_fact:
                facts[fact_id] = {
                    "content": fact_content,
                    "source": fact_source,
                    "added": fact_datetime_string,
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

        return self.vector_db.db["original_texts"].copy()

    def count_facts(self) -> int:
        """
        Returns the number of facts in the knowledge database.

        Returns:
            int: The number of facts in the knowledge database.
        """
        return self.vector_db.count()

    def get_fact_details(self, fact_id: str) -> dict:
        """
        Returns similar structure as query_to_fact_list() but for a specific fact ID. Returns NONE otherwise.

        Args:
            fact_id (str): The fact ID to get.

        Returns:
            dict: A dictionary with the content, source, added date, and added timestamp.
        """

        f_str = fact_id.lower()
        if f_str[0] == "f":
            f_id_as_int = int(f_str[1:])
        else:
            f_id_as_int = int(f_str)

        text, metadata = self.vector_db.get(f_id_as_int)

        fact_content = text
        fact_source = metadata["source"]
        fact_datetime = metadata["datetime"].strftime("%Y-%m-%d %H:%M:%S")
        fact_timestamp = metadata["added_on_timestamp"]

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

            result_ids = self.vector_db.query(query, n_results)

        else:

            result_ids = self.vector_db.query_min_date(
                query, since_date, n_results, "datetime"
            )

        facts = {}
        for f_id_as_int in result_ids:
            fact_id = f"f{f_id_as_int}"

            text, metadata = self.vector_db.get(f_id_as_int)

            fact_content = text
            fact_source = metadata["source"]
            fact_datetime = metadata["datetime"].strftime("%Y-%m-%d %H:%M:%S")
            fact_timestamp = metadata["added_on_timestamp"]

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

        f_str = fact_id.lower()
        if f_str[0] == "f":
            f_id_as_int = int(f_str[1:])
        else:
            f_id_as_int = int(f_str)

        text, metadata = self.vector_db.get(f_id_as_int)

        return metadata["source"]

    def get_fact_content(self, fact_id: str) -> str:
        """
        Returns the content of a fact given its ID.

        Args:
            fact_id: The fact ID to get.

        Returns:
            str: The content of the fact.
        """

        f_str = fact_id.lower()
        if f_str[0] == "f":
            f_id_as_int = int(f_str[1:])
        else:
            f_id_as_int = int(f_str)

        text, metadata = self.vector_db.get(f_id_as_int)

        return text

    def add_fact(self, fact: str, url: str) -> bool:
        """
        Adds a fact to the knowledge base.

        Args:
            fact (str): The fact to add.
            url (str): The URL source of the fact.

        Returns:
            bool: True if the fact was added, False otherwise.
        """
        return self.add_facts([fact], [url])

    def add_facts(self, facts: list, sources: list) -> bool:
        """
        Adds a facts to the knowledge base.

        Args:
            facts (list): List of strings. Each string is a fact.
            sources (list): List of sources (e.g., URLs) for the facts.

        Returns:
            bool: True if the facts were added, False otherwise.
        """

        # fact_id_start = self.facts_rag_collection.count() + 1

        added_now = datetime.now()
        added_now_timestamp = added_now.timestamp()
        # added_now_string = added_now.strftime("%Y-%m-%d %H:%M:%S")

        # fact_ids = []
        metadatas = []
        for i in range(0, len(facts)):
            # fact_ids.append(f"f{fact_id_start + i}")
            metadatas.append(
                {
                    "added_on_timestamp": added_now_timestamp,
                    "datetime": added_now,
                    "source": sources[i],
                }
            )

        self.vector_db.add_texts(facts, metadatas)
        self.vector_db.save()

        return True

    def facts_from_url(self, url: str, topic: str) -> None:
        """
        Given a URL, extract facts from it and save them to our DB and the facts dictionary. Also returns the facts in an array, in case one wants to analyze new facts.

        Args:
            url (str): Location of the content.
            topic (str): a brief description of the research you are undertaking.
        """

        content = self.get(url)

        facts = self.chunker.chunk(content, topic)
        sources = []
        for f in facts:
            sources.append(url)

        if len(facts) > 0:
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
                # try:
                self.facts_from_url(url, topic)
                # except:
                #    print("Error; failed to get content from " + url)

    # Builds a fact base baed on news from Bing.
    def new_get_new_bing_news(
        self, api_key, subscription_endpoint, topic, queries
    ) -> None:
        """
        Uses Bing to get recent news on the queries and associated topics.

        Args:
            api_key (str): The Bing API key.
            subscription_endpoint (str): The Bing subscription endpoint.
            topic (str): a brief description of the research you are undertaking.
            queries (list[str]): A list of queries to search for.
        """

        news_agent = NewsBingAgent(api_key, subscription_endpoint)
        for q in queries:
            results_urls = news_agent.get_news_as_list(q)
            for url in results_urls:
                if not self.in_cache(url):
                    print("NEWS RESULT: " + url)
                    self.facts_from_url(url, topic)

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

            try:
                results = webagent.search_google(
                    query=google_search_query,
                    custom_search_engine_id=self.google_search_id,
                    num=_DEFAULT_NUM_SEARCH_RESULTS,
                )
            except:
                print("Error; failed to get content from " + google_search_query)
                continue

            try:
                for result in results:
                    pass
            except:
                print(
                    "Error; failed to get 'result' in 'results' from "
                    + google_search_query
                )
                continue

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
                    fact_text = self.knowledge_db.get_fact_content(ref)
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
