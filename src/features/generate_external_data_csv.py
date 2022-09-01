import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import sessionmaker

'''
:description Script which generates a csv file containing all websites for which data has been collected.

'''

# connect to database
engine = create_engine(
    "sqlite:///../../data/processed/test_train_database.db", echo=True
)
sqlite_connection = engine.connect()
Base = automap_base()
Base.prepare(engine, reflect=True)
Rows = Base.classes.DryCleaned_EntirePage
Session = sessionmaker(bind=engine)
session = Session()

# Write results to csv
pd.DataFrame(session.query(Rows.website).distinct().all(), columns=["websites"]).to_csv(
    "../../data/external/raw_website_data.csv"
)
