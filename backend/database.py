"""
PostgreSQL access for VeritAI. Set DATABASE_URL, e.g.:
postgresql://veritai:veritai@localhost:5432/veritai
"""
from __future__ import annotations

import os
import logging
from contextlib import contextmanager
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


def database_url() -> str:
    url = (os.getenv("DATABASE_URL") or "").strip()
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Example: postgresql://user:pass@localhost:5432/veritai"
        )
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://") :]
    return url


@contextmanager
def connection(dict_rows: bool = True):
    conn = psycopg2.connect(database_url())
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _cursor(conn, dict_rows: bool):
    if dict_rows:
        return conn.cursor(cursor_factory=RealDictCursor)
    return conn.cursor()


def init_db() -> None:
    ddl = """
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        email VARCHAR(255) UNIQUE,
        phone VARCHAR(40) UNIQUE,
        password_hash TEXT,
        google_sub VARCHAR(255) UNIQUE,
        display_name VARCHAR(255),
        created_at TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS user_genres (
        user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        genre VARCHAR(64) NOT NULL,
        PRIMARY KEY (user_id, genre)
    );

    CREATE TABLE IF NOT EXISTS pending_otps (
        id SERIAL PRIMARY KEY,
        channel VARCHAR(16) NOT NULL,
        address VARCHAR(320) NOT NULL,
        code VARCHAR(16) NOT NULL,
        password_hash TEXT,
        expires_at TIMESTAMPTZ NOT NULL,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        UNIQUE (channel, address)
    );

    CREATE INDEX IF NOT EXISTS idx_pending_otps_exp ON pending_otps (expires_at);

    CREATE TABLE IF NOT EXISTS search_history (
        id SERIAL PRIMARY KEY,
        user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        input_mode VARCHAR(16),
        input_preview TEXT,
        verdict VARCHAR(32),
        confidence REAL,
        certainty VARCHAR(32),
        created_at TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_search_history_user ON search_history (user_id, created_at DESC);
    """
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
    logger.info("PostgreSQL schema ensured.")


def purge_expired_otps() -> None:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM pending_otps WHERE expires_at < NOW()")


def upsert_otp(
    channel: str,
    address: str,
    code: str,
    password_hash: str | None,
    ttl_seconds: int = 600,
) -> None:
    purge_expired_otps()
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO pending_otps (channel, address, code, password_hash, expires_at)
                VALUES (%s, %s, %s, %s, NOW() + (interval '1 second' * %s))
                ON CONFLICT (channel, address) DO UPDATE SET
                    code = EXCLUDED.code,
                    password_hash = EXCLUDED.password_hash,
                    expires_at = EXCLUDED.expires_at,
                    created_at = NOW()
                """,
                (channel, address, code, password_hash, ttl_seconds),
            )


def verify_otp_row(channel: str, address: str, otp: str) -> dict[str, Any] | None:
    purge_expired_otps()
    with connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT * FROM pending_otps
                WHERE channel = %s AND address = %s AND code = %s AND expires_at > NOW()
                """,
                (channel, address, otp),
            )
            row = cur.fetchone()
            if row:
                cur.execute(
                    "DELETE FROM pending_otps WHERE channel = %s AND address = %s",
                    (channel, address),
                )
            return dict(row) if row else None


def save_search_history(
    user_id: int,
    input_mode: str,
    input_preview: str,
    verdict: str,
    confidence: float,
    certainty: str,
) -> None:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO search_history
                    (user_id, input_mode, input_preview, verdict, confidence, certainty)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (user_id, input_mode, input_preview[:2000], verdict, confidence, certainty),
            )


def fetch_history(user_id: int, limit: int = 50) -> list[dict[str, Any]]:
    with connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, input_mode, input_preview, verdict, confidence, certainty, created_at
                FROM search_history
                WHERE user_id = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (user_id, limit),
            )
            return [dict(r) for r in cur.fetchall()]
