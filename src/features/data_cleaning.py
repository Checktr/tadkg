import re

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src import config

invalid_url_extensions = ["pdf", "ppt", "pptx", "xls", "xlsx"]


def clean_from_urls(
        url_list: list, sentence_split: bool, entity_name: str
) -> pd.DataFrame:
    """
    Collects data from a list of URLs and converts all sentences from websites into a Pandas Dataframe

    :param entity_name: name of entity data relates to
    :param sentence_split: if the sanitized data should be split by sentence or not
    :param url_list: list of URLs to process
    :return: pd.Dataframe - Dataframe containing the sanitized sentences
    """

    # Set user-agent to mozilla to avoid being forbidden from websites
    user_agent = {"User-agent": "Mozilla/5.0"}

    # Create the overall returned dataframe
    sanitized_data_df = pd.DataFrame(
        {},
        columns=['website', 'text', *list(config['data']['attributes'].keys())],
    )

    # Set Pandas Options for nicer debugging Print statements
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    pd.set_option("expand_frame_repr", False)

    for url in url_list:

        # TODO: accommodate for binary data like PDF files
        try:
            webpage = requests.get(url, headers=user_agent)
        except Exception:
            print(f"Invalid Webpage: {url}")
            continue

        if webpage is None:
            print(
                f'Invalid webpage response for URL: {url}'
            )
            continue

        if webpage.headers is None:
            print(
                f'Invalid webpage headers for URL: {url}'
            )
            continue

        if 'Content-Type' not in webpage.headers or webpage.headers["Content-Type"] is None:
            print(
                f'Content-Type Error for URL: {url}'
            )
            continue

        # Remove non-html pages
        if not webpage.headers["Content-Type"].startswith("text/html"):
            print(
                f'Invalid Content-Type {webpage.headers["Content-Type"]} for URL {url}'
            )
            continue

        # If the webpage forbids access, skip it
        if webpage.ok:

            soup = BeautifulSoup(webpage.content, features="html.parser")

            try:
                if sentence_split:
                    # Split raw text up by period followed by a space or double new line characters (possibly separated
                    # by a space)
                    raw_text_list = re.split(
                        "\. |(?:\n+ ?)+",
                        soup.get_text().replace("\t", "").replace("\r", ""),
                    )
                    stripped_text_list = [st.strip() for st in raw_text_list]
                    df = pd.DataFrame(stripped_text_list, columns=["text"])

                    # Remove input which is less than 3 words as it is not enough to distinguish a merger or acquisition
                    df = df[~df["text"].str.split().str.len().lt(3)]

                    # Populate the additional fields
                    for col in list(config['data']['attributes'].keys()):
                        df[col] = float("NaN")
                    df["website"] = url
                    df["entity"] = entity_name

                    # append data to dataframe
                    sanitized_data_df = pd.concat(
                        (sanitized_data_df, df), ignore_index=True
                    )
                else:
                    text = re.sub(r"[\n ]+", " ", soup.get_text()).strip()
                    d = {
                        "text": text,
                        "website": url,
                        "entity": entity_name
                    }
                    for col in list(config['data']['attributes'].keys()):
                        d[col] = float("NaN")

                    sanitized_data_df = pd.concat(
                        (
                            sanitized_data_df,
                            pd.DataFrame([d], columns=d.keys())
                        ),
                        ignore_index=True,
                    )

            except Exception as e:
                print("Error in data sanitization for ", url, "\n", e)

    return sanitized_data_df


def main():
    """
    Function Demonstrates the functionality of the dry_clean_the_data method on a test set of URLs

    :return: None
    """
    url_list = [
        "https://finance.yahoo.com/news/robust-demand-favorable-mix-aid-143502985.html",
        "https://www.nasdaq.com/articles/onsemi-on-to-report-q4-earnings%3A-whats-in-the-cards",
        "https://www.fool.com/earnings/call-transcripts/2022/02/07/on-semiconductor-on-q4-2021-earnings-call-transcri/",
        "https://www.marketscreener.com/quote/stock/ON-SEMICONDUCTOR-CORPORAT-10340/news/onsemi-Reports-Fourth"
        "-Quarter-and-2021-Financial-Results-37803051/",
        "https://m.marketscreener.com/quote/stock/ON-SEMICONDUCTOR-CORPORAT-10340/news/ON-SEMICONDUCTOR-CORP"
        "-Management-s-Discussion-and-Analysis-of-Financial-Condition-and-Results-of-Op-39412061/",
        "https://www.businesswire.com/news/home/20211101005252/en/onsemi-Completes-Acquisition-of-GT-Advanced"
        "-Technologies",
        "https://www.businesswire.com/news/home/20210825005781/en/onsemi-to-Acquire-GT-Advanced-Technologies",
        "https://evertiq.com/design/50890",
        "https://www.ept.ca/2021/08/on-semi-to-acquire-gt-advanced-technologies/",
        "https://www.ept.ca/2021/08/on-semi-to-acquire-gt-advanced-technologies/",
        "https://www.allaboutcircuits.com/news/onsemi-recalibrates-its-mission-through-rebranding-and-a-sic"
        "-acquisition/ "
    ]

    print(clean_from_urls(url_list, True, ''))


if __name__ == "__main__":
    main()
