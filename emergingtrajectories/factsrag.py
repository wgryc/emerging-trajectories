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

from datetime import datetime

from . import Client
from .crawlers import crawlerPlaywright
from .prompts import *
from .news import NewsAPIAgent

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

Please do not include new lines between bullet points.

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
        sources_file: str = "sources.json",
        facts_file: str = "facts.json",
        rag_db_folder="cdb",
        crawler=None,
    ) -> None:
        """
        This is a RAG-based fact database. We build a database of facts available in JSON and via RAG and use this as a basic search engine for information. We use ChromaDB to index all facts, but also maintain a list of facts, sources, etc. in a JSON file. Finally, we keep a cache of all content and assume URLs do not get updated; we'll change this process in the future.

        Args:
            folder_path (str): The folder where everything will be stored.
            openai_api_key (str): The OpenAI API key. Used for RAG embeddings.
            cache_file (str, optional): The name of the cache file. Defaults to "cache.json".
            sources_file (str, optional): The name of the sources file. Defaults to "sources.json".
            facts_file (str, optional): The name of the facts file. Defaults to "facts.json".
            rag_db_folder (str, optional): The folder where the ChromaDB database will be stored. Defaults to "cdb".
            crawler (optional): The crawler to use. Defaults to None, in which case a Playwright crawler will be used.
        """
        self.root_path = folder_path
        self.root_parsed = os.path.join(folder_path, "parsed")
        self.root_original = os.path.join(folder_path, "original")
        self.cache_file = os.path.join(folder_path, cache_file)
        self.sources_file = os.path.join(folder_path, sources_file)
        self.facts_file = os.path.join(folder_path, facts_file)
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

        # Set up / load facts dictionary
        # TODO Eventually, move this to a database or table or something.
        self.facts = self.load_facts()

        # Set up / load sources dictionary
        # TODO Eventually, move this to a database or table or something.
        self.sources = self.load_sources()

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

        r = []
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

        if len(r) == 0:
            return ""

        fact_content = ""
        if not skip_separator:
            fact_content = """--- START FACTS ---------------------------\n"""

        for item in r["ids"][0]:
            fact_content += item + ": " + self.facts[item]["content"] + "\n"

        if not skip_separator:
            fact_content += """--- END FACTS ---------------------------\n"""

        return fact_content

    def save_facts_and_sources(self) -> None:
        """
        Saves facts and sources to their respective files.
        """
        with open(self.facts_file, "w") as f:
            json.dump(self.facts, f, indent=4, cls=DjangoJSONEncoder)
        with open(self.sources_file, "w") as f:
            json.dump(self.sources, f, indent=4, cls=DjangoJSONEncoder)

    def facts_from_url(self, url: str, topic: str) -> list[str]:
        """
        Given a URL, extract facts from it and save them to ChromaDB and the facts dictionary. Also returns the facts in an array, in case one wants to analyze new facts.

        Args:
            url (str): Location of the content.

        Returns:
            list[str]: A list of facts extracted from the content.
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

        fact_id_start = self.facts_rag_collection.count() + 1

        for line in lines:
            if line[0:4] == "--- ":
                fact = line[4:]
                self.facts_rag_collection.add(
                    documents=[fact],
                    ids=[f"f{fact_id_start}"],
                    metadatas=[{"added_on_timestamp": datetime.now().timestamp()}],
                )

                self.facts[f"f{fact_id_start}"] = {
                    "added": datetime.now(),
                    "source": url,
                    "content": fact,
                    "cid": f"f{fact_id_start}",
                }

                fact_id_start += 1

        self.save_facts_and_sources()

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
                print("NEWS RESULT: " + result["url"])
                self.facts_from_url(result["url"], topic)

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
                        page_content = self.get(result.url)
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

    def load_facts(self) -> dict:
        """
        Loads the facts from the facts file.
        """
        if not os.path.exists(self.facts_file):
            with open(self.facts_file, "w") as f:
                f.write("{}")

        with open(self.facts_file, "r") as f:
            self.facts = json.load(f)

        return self.facts

    def load_sources(self) -> dict:
        """
        Loads the sources from the sources file.
        """
        if not os.path.exists(self.sources_file):
            with open(self.sources_file, "w") as f:
                f.write("{}")

        with open(self.sources_file, "r") as f:
            self.sources = json.load(f)

        return self.sources

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
        if fact_id in self.knowledge_db.facts:
            return self.knowledge_db.facts[fact_id]["source"]
        else:
            raise ValueError(f"Fact ID {fact_id} not found in the knowledge database.")


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
