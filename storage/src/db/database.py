from pathlib import Path
import sqlite3
from contextlib import contextmanager
from typing import Iterator, List, Dict, Any


class Database:
    def __init__(self, init_path: str | Path, db_path: str | Path):
        self.db_path = db_path
        self.init_path = init_path
        self._init_base()

    def _init_base(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            with open(self.init_path, "r", encoding="utf-8") as init_file:
                init_script = init_file.read()

            cursor.executescript(init_script)
            conn.commit()

    @contextmanager
    def get_connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def get(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def execute(self, query: str, params: tuple = ()) -> None:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()

    def truncate_table(self, table_name: str, cascade: bool = False) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute(f"SELECT COUNT(*) as count FROM {table_name}")
            count_before = cursor.fetchone()['count']
            
            if count_before == 0:
                return 0
            
            foreign_keys_state = None
            if cascade:
                cursor.execute("PRAGMA foreign_keys")
                foreign_keys_state = cursor.fetchone()[0]
                cursor.execute("PRAGMA foreign_keys = OFF")
            
            try:
                cursor.execute(f"DELETE FROM {table_name}")
                conn.commit()
                
                cursor.execute(f"DELETE FROM sqlite_sequence WHERE name='{table_name}'")
                conn.commit()
                
                if cascade and foreign_keys_state:
                    cursor.execute(f"PRAGMA foreign_keys = {foreign_keys_state}")
                
                return count_before
                
            except sqlite3.Error as e:
                conn.rollback()
                raise Exception(f"Ошибка при очистке таблицы {table_name}: {e}")

