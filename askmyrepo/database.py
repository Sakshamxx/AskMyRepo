"""The read-only SQL tool: one SELECT on ShopLite's database, with a row limit and a timeout."""

import sqlite3
import time
from pathlib import Path

from askmyrepo.tools import ToolError

MAX_ROWS = 50
TIMEOUT_S = 2.0
# What a query may do (sqlite3 authorizer action codes): read tables and views, call functions,
# and use a recursive WITH. Everything else (writes, ATTACH, PRAGMA...) is denied.
ALLOWED_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION,
                   sqlite3.SQLITE_RECURSIVE}


# Layer 1+2 of the SQL guard: read-only file plus an authorizer.
def connect_read_only(db_path: Path) -> sqlite3.Connection:
    """Open the database read-only, with an authorizer that denies everything except reading."""
    if not db_path.exists():
        raise FileNotFoundError(f"no database at {db_path}: run ShopLite's scripts/seed.py")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    conn.set_authorizer(lambda action, *args: sqlite3.SQLITE_OK if action in ALLOWED_ACTIONS
                        else sqlite3.SQLITE_DENY)
    return conn


# Layer 3: one SELECT only, with a row cap and a time limit.
def run_select(conn: sqlite3.Connection, sql: str, max_rows: int = MAX_ROWS,
               timeout_s: float = TIMEOUT_S) -> dict:
    """Run one SELECT; return its columns and at most max_rows rows, or raise ToolError."""
    statement = sql.strip().rstrip(";").strip()
    if not statement.lower().startswith(("select", "with")):
        raise ToolError("only a single SELECT (or WITH ... SELECT) query is allowed")
    if ";" in statement:
        raise ToolError("one statement only: remove the extra ';'")
    deadline = time.monotonic() + timeout_s
    # Called every 10,000 SQLite steps; returning True stops the query.
    conn.set_progress_handler(lambda: time.monotonic() > deadline, 10_000)
    try:
        cursor = conn.execute(statement)
        rows = cursor.fetchmany(max_rows + 1)
    except sqlite3.OperationalError as exc:
        if "interrupted" in str(exc):
            message = f"the query took longer than {timeout_s:.0f} s and was stopped"
            raise ToolError(message) from exc
        raise ToolError(f"SQLite error: {exc}") from exc
    except sqlite3.DatabaseError as exc:  # includes "not authorized"
        raise ToolError(f"SQLite refused the query: {exc}") from exc
    finally:
        conn.set_progress_handler(None, 0)
    columns = [d[0] for d in cursor.description]
    return {"columns": columns, "rows": [list(row) for row in rows[:max_rows]],
            "truncated": len(rows) > max_rows}
