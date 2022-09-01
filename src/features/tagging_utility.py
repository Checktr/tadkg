"""
tagging_utility.py

Used to manually label data within the test_train database. Cycles through random rows
of the database (user must choose whether to label pages or sentences) and prompts the user
to answer whether the row represents a merger, acquisition, both or neither, then saves results
to the database.
"""

import webbrowser
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql.expression import func

from src import config
from src.features import data_cleaning

hotkeys = ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l']
tag_options = config["tagging"]["options"]

# Connect to the database
db_path = f'sqlite:///{Path(Path(__file__).parent.parent.parent, config["paths"]["databases"]["test_train"])}'
engine = create_engine(db_path, echo=True)
sqlite_connection = engine.connect()
Base = automap_base()
Base.prepare(engine, reflect=True)

Rows = None
is_sentence = False

# Set "Rows" to target the "DryCleaned" or "DryCleaned_EntirePage" table depending on whether pages or sentences are
# being labelled
while Rows is None:
    print("\n\nAre you labelling an entire page or sentences?")
    print("p - page")
    print("s - sentences")
    text_in = input()

    print()
    if text_in == "p":
        Rows = Base.classes[config['data']['tables']['page_level']]

    elif text_in == "s":
        Rows = Base.classes[config['data']['tables']['sentence_level']]
        is_sentence = True

print("\n")

# More database setup
Session = sessionmaker(bind=engine)
session = Session()

# Only label rows with null values in all labelling columns
for row in (
        session.query(Rows)
                .filter(Rows.isMergerOrAcquisition.is_(None))
                .filter(Rows.isMerger.is_(None))
                .filter(Rows.isAcquisition.is_(None))
                .order_by(func.random())
):
    while True:
        print(f"\n\nRow {row.index}\n")

        # Print row data
        if is_sentence:
            print(f"TEXT: {row.text}\n")
        else:
            print(f"URL:\n{row.website}\n")
            chunks = [row.text[i: i + 256] for i in range(0, len(row.text), 256)]
            # Text preview
            print("TEXT:")
            if len(chunks) < 6:
                print("\n".join(chunks))
            else:
                print("\n".join(chunks[:6]))
            lowercase = row.text.lower()
            if (
                    "adblock" in lowercase
                    or "javascript" in lowercase
                    or "internet explorer" in lowercase
                    or "cookies" in lowercase
            ):
                # Search for common terms that might indicate a failure to load the webpage.
                # JavaScript rendering moment!!!
                print(
                    "\n!! Detected a term that might indicate a failure to render. Check the text above to ensure it "
                    "matches the browser."
                )
                print('   If it does not match, mark "neither" !!')

            # Open webpages in the default browser to manually check them
            webbrowser.open(row.website)

        print("\nWhat kind of data is this?")
        for i in range(len(tag_options)):
            print(f'{hotkeys[i]} - {tag_options[i]["name"]}')
        print('q - Quit')
        print("")
        in_char = input("")

        check_values = [0, 0, 0]

        # Label data based on user input
        if in_char in hotkeys and hotkeys.index(in_char) < len(tag_options):
            check_values = tag_options[hotkeys.index(in_char)]['values']
        elif in_char == "q":
            print("Quitting")
            exit(0)
        else:
            print("Invalid")
            continue

        # Apply to database
        row.isMerger = check_values[1]
        row.isAcquisition = check_values[0]
        row.isMergerOrAcquisition = check_values[2]
        session.commit()

        # If a page has been marked as a merger or acquisition, parse it into sentences and add the results
        # to the "sentences" table to be labelled individually.
        if check_values[2] == 1 and not is_sentence:
            print("Importing page to sentence database...")
            prepared_df = data_cleaning.clean_from_urls(
                [row.website], True, row.entity
            )
            prepared_df.to_sql(
                config['data']['tables']['sentence_level'], sqlite_connection, if_exists="append", index=False
            )

        break
