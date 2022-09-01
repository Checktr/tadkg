import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path

from src import config
from src.features import data_cleaning

db_columns = [
    config['synthetic_page_schema']['attribute'],
    config['synthetic_page_schema']['target'],
    config['synthetic_page_schema']['entity']
]

for key in list(config['synthetic_page_schema']['defaults'].keys()):
    db_columns.append(key)

page_df = pd.DataFrame({}, columns=db_columns, )

i = 0
with open(config['synthetic_data']['negative_page_file_name'], 'r') as file:
    for line in file:
        if i <= 102:
            i += 1
            continue
        if i > 152:
            break
        url = str(line.split(' ')[-1][:-1])
        print(url)
        name = ' '.join(line.split(' ')[:-1])
        print(name)
        try:
            df = data_cleaning.clean_from_urls([url], False, name)
            for key in list(config['synthetic_page_schema']['defaults'].keys()):
                value = config['synthetic_page_schema']['defaults'][key]
                if value is not None:
                    df[key][0] = value
                else:
                    df[key][0] = float("NaN")

            page_df = pd.concat([page_df, df])
            i += 1
        except Exception:
            print(f'Error parsing {name}')

engine = create_engine(
    f'sqlite:///{Path(Path(__file__).parent.parent.parent.parent, config["paths"]["databases"]["test_train"])}',
    echo=True
)
connection = engine.connect()

page_df.to_sql(config['data']['tables']['page_level'], connection, if_exists="append", index=False)
