from pathlib import Path
import sqlite3
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

DB_PATH = Path("../shared/db/data.db").resolve().absolute()
INIT_FILE = Path("../shared/db/init.sql").resolve().absolute()

def init_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    with open(INIT_FILE, 'r', encoding='utf-8') as f:
            sql_script = f.read()
        
    cursor.executescript(sql_script)

    conn.commit()
    conn.close()

class User(BaseModel):
    name: str
    email: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Инициализация при старте
    init_database()
    yield
    # Очистка при завершении
    pass

app = FastAPI(lifespan=lifespan)

@app.post("/users/")
def create_user(user: User):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO users (name, email) VALUES (?, ?)', 
                   (user.name, user.email))
    conn.commit()
    user_id = cursor.lastrowid
    conn.close()
    return {"id": user_id, **user.dict()}

@app.get("/users/")
def get_users():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users')
    users = cursor.fetchall()
    conn.close()
    return {"users": [{"id": u[0], "name": u[1], "email": u[2]} for u in users]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

