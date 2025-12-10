import json
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List

from pydantic import BaseModel
from fastapi import FastAPI, Request, Query
import uvicorn

from db import Database


logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


DB_PATH = Path("../shared/db/data.db").resolve().absolute()
INIT_FILE = Path("../shared/db/init.sql").resolve().absolute()

database = None


class StandData(BaseModel):
    """Stand metadata for creation and management."""

    name: str
    description: str
    section: str


class VisitData(BaseModel):
    """Visitor data for analytics and storage."""

    gender: str
    group: str
    age_group: str
    age: float
    name: str
    datetime: str
    time_elapsed: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and tear down shared application resources."""

    global database
    
    print(f"Initializing database from: {INIT_FILE}")
    print(f"Database path: {DB_PATH}")
    
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    database = Database(INIT_FILE, DB_PATH)
    
    try:
        test_result = database.get("SELECT 1 as test")
        print(f"Database connected successfully: {test_result}")
    except Exception as e:
        print(f"Database connection error: {e}")

    config_path = Path("../shared/examples/sections.json").absolute().resolve()

    sections = {
        "default": {"description": "Not associated stands"}
    }
    if config_path.exists():
        with open(config_path, "r") as config_file:
            sections.update(json.load(config_file))

    for label, desc in sections.items():
        database.execute('''INSERT INTO sections (label, description)
                            VALUES (?, ?)
                            ON CONFLICT(label) DO NOTHING''', (label, desc["description"]))
    
    yield
    
    print("Shutting down database...")


app = FastAPI(
    title="Museum Assistant API",
    description="API для управления музеем и посетителями",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
def read_root():
    """Return API info and main endpoints."""

    return {
        "message": "Museum Assistant API",
        "endpoints": {
            "stands": "/stands/",
            "visits": "/visits/"
        }
    }

@app.get("/health")
def health_check():
    """Check API and database health status."""

    global database
    try:
        database.get("SELECT 1")
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.get("/sections/")
def get_sections() -> List[dict]:
    """Return all sections ordered by id."""

    global database
    sections = database.get("SELECT * FROM sections ORDER BY id")
    return sections

@app.get("/sections/clear")
def clear_sections() -> None:
    """Remove all sections from the database."""

    global database
    database.truncate_table("sections")

@app.get("/stands/")
def get_stands() -> List[dict]:
    """Return all stands ordered by id."""

    global database
    stands = database.get("SELECT * FROM stands ORDER BY id")
    return stands


@app.get("/stands/{stand_id}")
def get_stand(stand_id: int) -> dict:
    """Return stand details by numeric id."""

    global database
    result = database.get("SELECT * FROM stands WHERE id = ?", (stand_id,))
    if result:
        return result[0]
    return {"error": "Stand not found"}


@app.get("/visits/")
def get_visits() -> List[dict]:
    """Return all visits ordered by timestamp."""

    global database
    visits = database.get("SELECT * FROM visits ORDER BY timestamp DESC")
    return visits


@app.get("/visits/{visit_id}")
def get_visit(visit_id: int) -> dict:
    """Return visit details by numeric id."""

    global database
    result = database.get("SELECT * FROM visits WHERE id = ?", (visit_id,))
    if result:
        return result[0]
    return {"error": "Visit not found"}

@app.get("/api/visits/all")
def get_all_visits() -> List[dict]:
    """Return all visits without filters."""

    global database
    all_visits = database.get('''SELECT * FROM visits
                                 ORDER BY timestamp DESC''')
    return all_visits

@app.get("/api/stands/{stand_name}/date_range")
def get_dates_range_by_stand(stand_name: str):
    """Return all visit timestamps for a stand."""

    global database
    stand_id = database.get("SELECT id FROM stands WHERE name = ?", (stand_name,))
    if not stand_id:
        return []
    
    dates = database.get('''SELECT
                                MIN(DATE(timestamp)) as min_date,
                                MAX(DATE(timestamp)) as max_date
                            FROM visits
                            WHERE stand_id = ?''', (stand_id[0]["id"],))
    return dates[0]

@app.get("/api/stands/{stand_name}/dates")
def get_dates_range_by_stand(stand_name: str):
    """Return all visit timestamps for a stand ordered by time descending."""
    
    global database
    stand_id = database.get("SELECT id FROM stands WHERE name = ?", (stand_name,))
    if not stand_id:
        return []

    dates = database.get("SELECT timestamp FROM visits WHERE stand_id = ? ORDER BY timestamp DESC", (stand_id[0]["id"],))
    
    return dates

@app.get("/api/stands/{stand_name}/stats")
def get_stats_by_stand(stand_name: str,
                       start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
                       end_date: str = Query(..., description="End date (YYYY-MM-DD)")):
    """Return aggregated visit stats for a stand and date range."""

    global database
    stand_id = database.get("SELECT id FROM stands WHERE name = ?", (stand_name,))
    if not stand_id:
        return []

    stand_id = stand_id[0]["id"]
    stats = database.get('''SELECT
                                AVG(age) as avg_age,
                                AVG(time_elapsed) as avg_time_elapsed,
                                COUNT (*) as total_visits,
                                (
                                    SELECT age_group 
                                    FROM visits 
                                    WHERE stand_id = ? 
                                    AND DATE(timestamp) BETWEEN ? AND ?
                                    GROUP BY age_group 
                                    ORDER BY COUNT(*) DESC 
                                    LIMIT 1
                                ) as most_common_age_group,
                                (
                                    SELECT gender 
                                    FROM visits 
                                    WHERE stand_id = ? 
                                    AND DATE(timestamp) BETWEEN ? AND ?
                                    GROUP BY gender 
                                    ORDER BY COUNT(*) DESC 
                                    LIMIT 1
                                ) as most_common_gender
                            FROM visits
                            WHERE stand_id = ?
                                AND DATE(timestamp) BETWEEN ? AND ?''', (stand_id, start_date, end_date,
                                                                         stand_id, start_date, end_date,
                                                                         stand_id, start_date, end_date))
    return stats[0]

@app.get("/api/stands/names")
def get_stands_names() -> List[dict]:
    """Return all stand names."""

    global database
    all_names = database.get("SELECT name FROM stands ORDER BY id DESC")
    return all_names

@app.post("/api/stands/push")
async def push_stand(data: StandData, request: Request):
    """Create a new stand linked to a section."""

    global database
    database.execute('''INSERT INTO stands (section_id, name, description)
                        SELECT
                            sec.id,
                            ?,
                            ?
                        FROM sections sec
                        WHERE sec.label = ?''', (data.name, data.description, data.section))
    print(data)

@app.post("/api/visits/push")
async def push_visit(data: VisitData, request: Request):
    """Store a new visit for a stand."""

    global database
    database.execute('''INSERT INTO visits (stand_id, gender, age_group, age, timestamp, time_elapsed)
                        SELECT
                            stnd.id,
                            ?,
                            ?,
                            ?,
                            ?,
                            ?
                        FROM stands stnd
                        WHERE stnd.name = ?''', (data.gender, data.age_group, data.age, data.datetime, data.time_elapsed, data.name))
    print(data)


if __name__ == "__main__":
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
        )
    except Exception as err:
        logging.error(err)

