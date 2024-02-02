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

"""
CACHE STRUCTURE IN JSON...

key: URI (URL or file name)
value: {
    "obtained_on": <date>; when the file was downloaded
    "last_accessed": <date>; when the file was last used by the agent
}

"""

def uri_to_local(uri):
    uri_md5 = hashlib.md5(uri.encode('utf-8')).hexdigest()
    return uri_md5

class KnowledgeBaseFileCache:

    def __init__(self, folder_path, cache_file="cache.json"):
        self.root_path = folder_path
        self.root_parsed = os.path.join(folder_path, "parsed")
        self.root_original = os.path.join(folder_path, "original")
        self.cache_file = os.path.join(folder_path, cache_file)
        self.cache = self.load_cache()

    def load_cache(self):

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

    def in_cache(self, uri):
        if uri in self.cache:
            return True
        return False

    def update_cache(self, uri, obtained_on, last_accessed):
        uri_md5 = uri_to_local(uri)
        self.cache[uri] = {
            "obtained_on": obtained_on,
            "last_accessed": last_accessed,
            "uri_md5": uri_md5
        }
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f, cls=DjangoJSONEncoder)

    def get(self, uri):
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