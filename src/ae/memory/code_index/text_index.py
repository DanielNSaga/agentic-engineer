"""Full-text search index backed by SQLite FTS5."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List


class TextIndex:
    """SQLite FTS5-backed full text index for source files."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def index_document(self, path: Path, content: str) -> None:
        path_key = self._path_key(path)
        with self._connect() as connection:
            connection.execute("DELETE FROM documents WHERE path = ?", (path_key,))
            connection.execute(
                "INSERT INTO documents(path, content) VALUES(?, ?)",
                (path_key, content),
            )

    def remove(self, path: Path) -> None:
        path_key = self._path_key(path)
        with self._connect() as connection:
            connection.execute("DELETE FROM documents WHERE path = ?", (path_key,))

    def search(self, query: str, limit: int = 5) -> List[str]:
        with self._connect() as connection:
            cursor = connection.execute(
                "SELECT path FROM documents WHERE documents MATCH ? LIMIT ?",
                (query, limit),
            )
            rows = cursor.fetchall()
        return [row[0] for row in rows]

    def list_paths(self) -> List[str]:
        with self._connect() as connection:
            cursor = connection.execute("SELECT path FROM documents ORDER BY path")
            rows = cursor.fetchall()
        return [row[0] for row in rows]

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS documents USING fts5(path UNINDEXED, content)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(
            self._db_path,
            detect_types=sqlite3.PARSE_DECLTYPES,
        )

    def _path_key(self, path: Path) -> str:
        return path.as_posix()
