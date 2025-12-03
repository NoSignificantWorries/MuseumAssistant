import json
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List

from pydantic import BaseModel
from fastapi import FastAPI, Request
import uvicorn

from db import Database

DB_PATH = Path("../shared/db/data.db").resolve().absolute()
INIT_FILE = Path("../shared/db/init.sql").resolve().absolute()

database = None


class StandData(BaseModel):
    name: str
    description: str
    section: str


class VisitData(BaseModel):
    gender: str
    group: str
    age_group: str
    age: float
    name: str
    datetime: str
    time_elapsed: float


@asynccontextmanager
async def lifespan(app: FastAPI):
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
    return {
        "message": "Museum Assistant API",
        "endpoints": {
            "stands": "/stands/",
            "visits": "/visits/"
        }
    }

@app.get("/sections/")
def get_sections() -> List[dict]:
    global database
    sections = database.get("SELECT * FROM sections ORDER BY id")
    return sections

@app.get("/sections/clear")
def clear_sections() -> None:
    global database
    database.truncate_table("sections")

@app.get("/stands/")
def get_stands() -> List[dict]:
    global database
    stands = database.get("SELECT * FROM stands ORDER BY id")
    return stands


@app.get("/stands/{stand_id}")
def get_stand(stand_id: int) -> dict:
    global database
    result = database.get("SELECT * FROM stands WHERE id = ?", (stand_id,))
    if result:
        return result[0]
    return {"error": "Stand not found"}


@app.get("/visits/")
def get_visits() -> List[dict]:
    global database
    visits = database.get("SELECT * FROM visits ORDER BY timestamp DESC")
    return visits


@app.get("/visits/{visit_id}")
def get_visit(visit_id: int) -> dict:
    global database
    result = database.get("SELECT * FROM visits WHERE id = ?", (visit_id,))
    if result:
        return result[0]
    return {"error": "Visit not found"}


@app.get("/health")
def health_check():
    global database
    try:
        database.get("SELECT 1")
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.post("/api/stands/push")
async def push_stand(data: StandData, request: Request):
    global database
    database.execute('''INSERT INTO stands (section_id, name, description)
                        SELECT
                            sec.id,
                            ?,
                            ?
                        FROM sections sec
                        WHERE sec.label = ?''', (data.name, data.description, data.section))
    print(data)

# INSERT INTO visits (stand_id, gender, age_group, age, timestamp, time_elapsed)
# SELECT 
#   s.id,
#   'male',
#   '25-40',
#   32.5,
#   CURRENT_TIMESTAMP,
#   12.3
# FROM stands s
# WHERE s.name = 'VR-зона';
@app.post("/api/visits/push")
async def push_visit(data: VisitData, request: Request):
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
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )

