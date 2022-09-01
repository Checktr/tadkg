"""
link_grabber.py

File used to search Google for company names and return bunches of URLs from the searches.
"""
from pathlib import Path

import pandas as pd
from googlesearch import search

from src import config

# Used to remove duplicate URLs
url_cache = []


# Get a list of URLs from a Google search query
def query_links(query: str):
    url_list = []
    try:
        for j in search(query, **config['collection']['google_search']):
            url_list.append(j)
    except Exception as e:
        print(f"Found an error when searching {query}")
        print(repr(e))

    return url_list


# Generate groups of links from company names in a csv file
def grab_csv_links(filename: Path, keywords: list):
    entities = pd.read_csv(filename)
    for data in grab_links(entities, keywords):
        yield data


# Generate groups of links from a Pandas DataFrame containing company names
def grab_links(entity_data, keywords):
    is_single_column_input = len(entity_data.columns) == 1
    for index, row in entity_data.iterrows():
        for kw in keywords:

            if is_single_column_input:
                r = row.get(0)
            else:
                r = '"' + row.get(0) + '" and "' + row.get(1) + '"'

            # Replace {} with the current keyword - eg. "{} merger" becomes "<COMPANY> merger"
            query_string = kw.replace("{}", r)
            print(f"Searching: '{query_string}'")
            raw_links = query_links(query_string)
            iter_links = []
            for link in raw_links:
                if link not in url_cache:
                    url_cache.append(link)
                    iter_links.append(link)
            yield {"entity": r, "links": iter_links}
            """
            wait_period = random.randint(5, 10)
            print("Sleeping for " + str(wait_period) + " seconds")
            time.sleep(wait_period)
            """
