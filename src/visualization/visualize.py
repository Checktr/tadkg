"""
visualize.py

Basic WSGI server used as the backend for the user interface and data visualization component.
Server is hosted on http://localhost:4096 by default.
Allows users to view data by company/webpage/sentence with NER labelling. Also allows users to add company names
and/or upload CSV files to be processed by the system and added to the live database.
"""

import os
import sqlite3

from bottle import route, run, static_file, request, redirect, template

from src.models.predict.real_world_detection import apply_data

# Initialize database
db_name = os.path.join("..", "..", "data", "live", "liveMergersAcquisitionsDB.db")
conn = sqlite3.connect(db_name)


# Default route - Display the dashboard
@route("/")
def hello():
    return static_file("index.html", "")


# Favicon
@route("/favicon.ico")
def favicon():
    return static_file("favicon.ico", "assets")


# Page to display when the user has confirmed they wish to add a company to the database via text or CSV file
@route("/add_company", method="POST")
def add_company():
    if "csv" in request.files:
        # Save the CSV file to data/raw/ and process it
        csv_file = request.files.get("csv")
        dl_path = os.path.realpath(
            os.path.join(
                os.path.dirname(__file__), "..", "..", "data", "raw", csv_file.filename
            )
        )
        csv_file.save(dl_path, overwrite=True)
        with open("add_company.html", "r") as html_file:
            return template(
                html_file.read(), name=csv_file.filename, plural=True, path=dl_path
            )
    else:
        # User entered a name; use that name and process it into the database
        with open("add_company.html", "r") as html_file:
            return template(
                html_file.read(),
                name=request.forms.get("name"),
                plural=False,
                path=None,
            )


# Assets directory
@route("/assets/<path:path>")
def asset(path):
    return static_file(path, "assets")


# Fetch data on a specific company and return it as a JSON object
@route("/api/fetch_data/<name>")
def api_name(name):
    print(name)
    cur = conn.cursor()
    # Grab a list of URLs associated with a company, plus their confidence scores and dates
    query = "SELECT DISTINCT website, company, confidenceScore, date_processed " \
            "FROM Webpage_Confidence_Rankings NATURAL JOIN Processed_Websites " \
            "WHERE company = ? ORDER BY confidenceScore DESC"
    cur.execute(query, [name])
    res = cur.fetchall()
    # Initialize the JSON object with company info
    data = [
        {"url": item[0], "confidenceScore": item[2], "dateProcessed": item[3]}
        for item in res
    ]
    for point in data:
        # For each URL, grab every sentence associated with it, along with the text's labels and confidence score
        query = "SELECT DISTINCT text, isMerger, isAcquisition, confidenceScore " \
                "FROM Processed_Sentences WHERE queriedCompany = ? AND website = ? " \
                "GROUP BY text ORDER BY confidenceScore DESC"
        cur.execute(query, [name, point["url"]])
        items = cur.fetchall()
        # Set "sentences" to an array of these sentence objects
        point["sentences"] = [
            {
                "text": item[0],
                "type": "either"
                if item[1] == item[2]
                else ("merger" if item[1] == "1" else "acquisition"),
                "confidenceScore": item[3],
            }
            for item in items
        ]

        for sentence in point["sentences"]:
            # For each sentence, fetch a list of NER-recognized entities and add it to the sentence object
            query = "SELECT DISTINCT involvedCompany FROM Processed_Sentences " \
                    "WHERE queriedCompany = ? AND website = ? AND text = ?"
            cur.execute(query, [name, point["url"], sentence["text"]])
            sentence["companies"] = [item[0] for item in cur.fetchall()]

    cur = conn.cursor()
    # Get the company's confidence score
    query = f"SELECT confidenceScore FROM Company_Confidence_Rankings WHERE company = ?"
    cur.execute(query, [name])
    res = cur.fetchone()

    # Return the JSON object
    return {
        "name": name,
        "confidenceScore": res[0] if res is not None else "?",
        "data": data,
    }


# Return a JSON object containing a list of companies with their names and confidence scores.
@route("/api/companies")
def api_companies():
    cur = conn.cursor()
    query = f"SELECT company, confidenceScore FROM Company_Confidence_Rankings"
    cur.execute(query)
    res = cur.fetchall()
    company_list = [{"name": r[0], "confidenceScore": r[1]} for r in res]
    return {"companies": company_list}


# Add a company based on a CSV path or a company name.
@route("/api/add_company")
def add_company():
    if "csv" in request.query:
        apply_data(filename=request.query.csv)
    else:
        apply_data(names=[request.query.name])
    redirect("/")


run(host="localhost", port=4906, debug=True)
