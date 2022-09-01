import asyncio
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String

from src import config
from src.features import data_cleaning, link_grabber

# If the script is passed a path to a CSV file, it will use that file for processing
filename = Path(Path(__file__).parent.parent.parent, config['paths']['entities'])

filtered_args = list(filter(lambda name: ".csv" in name, sys.argv))
if len(filtered_args) > 0:
    filename = Path(filtered_args[0])

columns = ['website', 'text', *list(config['data']['attributes'].keys())]
sanitized_data_df = pd.DataFrame(
    {},
    columns=columns,
)

# Open a SQLite connection to dump results into
db_path = f'sqlite:///{Path(Path(__file__).parent.parent.parent, config["paths"]["databases"]["test_train"])}'
engine = create_engine(db_path, echo=True)
sqlite_connection = engine.connect()

meta = MetaData()


def setup_table(name):
    data_cols = list(map(lambda name: Column(name, String), list(config['data']['attributes'].keys())))
    if not engine.dialect.has_table(sqlite_connection, name):
        Table(
            name,
            meta,
            Column("index", Integer, primary_key=True, autoincrement=True),
            Column("website", String),
            Column("text", String),
            Column("entity", String),
            *data_cols
        )


setup_table(config['data']['tables']['sentence_level'])
setup_table(config['data']['tables']['page_level'])

meta.create_all(engine)


# Function to dump new results into the SQLite database
async def concat_df(url_list: list, entity_name: str, table_name, connection):
    global sanitized_data_df
    new_sanitized = data_cleaning.clean_from_urls(url_list, False, entity_name)
    print(new_sanitized)
    new_sanitized.to_sql(
        table_name, connection, if_exists="append", index=False
    )


# For each list of URLs grabbed, dump the sanitized contents to the database
for link_data in link_grabber.grab_csv_links(filename, config['collection']['templates']):
    asyncio.run(
        concat_df(link_data["links"], link_data["entity"], config['data']['tables']['page_level'], sqlite_connection))
