from fastapi import FastAPI
from pydantic import BaseModel
import csv
import os

app = FastAPI(title="SAP TCode Search API")

DATA_FILE = "tcodes.csv"

# ---------- Load data at startup ----------
tcodes_data = []

def load_data():
    global tcodes_data
    if not os.path.exists(DATA_FILE):
        print("tcodes.csv NOT FOUND")
        return

    with open(DATA_FILE, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        tcodes_data = [row for row in reader]

    print(f"Loaded {len(tcodes_data)} tcodes")

load_data()


# ---------- Request Model ----------
class SearchRequest(BaseModel):
    query: str


# ---------- Health ----------
@app.get("/")
def root():
    return {"status": "API running", "records": len(tcodes_data)}


# ---------- Search ----------
@app.post("/search-tcode")
def search_tcode(request: SearchRequest):
    q = request.query.lower()

    results = []
    for row in tcodes_data:
        text = f"{row.get('tcode','')} {row.get('description','')}".lower()
        if q in text:
            results.append(row)

        if len(results) >= 5:
            break

    return {
        "query": request.query,
        "results": results
    }
