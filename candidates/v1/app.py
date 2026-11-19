import json
import os
import sqlite3
from contextlib import contextmanager
from typing import List

import uvicorn
from fastapi import FastAPI, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, create_model, field_validator

import aging_probe  # AGING-PROBE

DB_PATH = "db.sqlite3"

app = FastAPI(
    title="Credit Card Phone Association Service",
    description="A simple service to checking credit cards with phones.",
    version="1.0.0",
)

# If you ever need a secret, it's available here (not required for this implementation)
APP_SECRET = os.environ.get("APP_SECRET", "")


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS associations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                credit_card TEXT NOT NULL,
                phone TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(credit_card, phone)
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_assoc_phone ON associations(phone);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_assoc_card ON associations(credit_card);")
        # Covering index for /retrieve_cards: rows for a phone come out already ordered by
        # credit_card, with no table lookup and no temp B-tree. Additive; built once on an
        # existing db at startup.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_assoc_phone_card ON associations(phone, credit_card);"
        )
        conn.commit()
    finally:
        conn.close()


@contextmanager
def db_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys=ON;")
        yield conn
        conn.commit()
    finally:
        conn.close()


@app.on_event("startup")
def on_startup():
    init_db()
    aging_probe.start(DB_PATH)  # AGING-PROBE


@app.on_event("shutdown")  # AGING-PROBE
def _aging_probe_shutdown():  # AGING-PROBE
    aging_probe.stop()  # AGING-PROBE


# Return 400 (not 422) for invalid requests, per the schema
@app.exception_handler(RequestValidationError)
def validation_exception_handler(_request, _exc: RequestValidationError):
    aging_probe.event("validation_error", path=getattr(getattr(_request, "url", None), "path", ""))  # AGING-PROBE
    return JSONResponse(status_code=400, content={"detail": "Invalid request"})


def normalize_card(card: str) -> str:
    # Remove spaces and hyphens; keep other characters as-is
    return "".join(ch for ch in card.strip() if ch not in {" ", "-"})


def normalize_phone(phone: str) -> str:
    return phone.strip()


class AssociateCardRequest(BaseModel):
    credit_card: str = Field(..., description="Number of the credit card")
    phone: str = Field(..., description="Phone number")

    @field_validator("credit_card")
    @classmethod
    def validate_card(cls, v: str) -> str:
        v = normalize_card(v)
        if not v:
            raise ValueError("credit_card must be non-empty")
        # Basic sanity check: allow digits-only after normalization
        if not v.isdigit():
            raise ValueError("credit_card must contain only digits (spaces/hyphens allowed)")
        return v

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v: str) -> str:
        v = normalize_phone(v)
        if not v:
            raise ValueError("phone must be non-empty")
        return v


class RetrieveCardsRequest(BaseModel):
    phone_numbers: List[str] = Field(..., description="Phone numbers", min_length=1)

    @field_validator("phone_numbers")
    @classmethod
    def validate_phones(cls, v: List[str]) -> List[str]:
        phones = [normalize_phone(p) for p in v]
        if any(not p for p in phones):
            raise ValueError("phone_numbers must contain non-empty strings")
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for p in phones:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        if not unique:
            raise ValueError("phone_numbers must be non-empty")
        return unique


# OpenAPI-only model for the 200 body; same component name and schema as before.
RetrieveCardsResponseSchema = create_model(
    "RetrieveCardsResponse",
    card_numbers=(List[str], Field(..., description="Credit cards associated with all given phone numbers.")),
)

_JSON_CHUNK = 4096  # cards per json.dumps call, so no single call holds the GIL for long


class RetrieveCardsResponse(Response):
    """The 200 body, rendered once in the worker thread that runs the handler.

    Returning a Response makes FastAPI skip response_model validation/serialization and the
    JSON encoding on the event loop. Bytes and headers are identical to the JSONResponse that
    FastAPI built before: {"card_numbers":[...]} encoded with ensure_ascii=False and
    separators=(",", ":"), content-type application/json.
    """

    media_type = "application/json"

    def __init__(self, card_numbers: List[str]) -> None:
        out = [b'{"card_numbers":[']
        for i in range(0, len(card_numbers), _JSON_CHUNK):
            if i:
                out.append(b",")
            chunk = json.dumps(
                card_numbers[i:i + _JSON_CHUNK],
                ensure_ascii=False,
                allow_nan=False,
                indent=None,
                separators=(",", ":"),
            )
            out.append(chunk[1:-1].encode("utf-8"))  # drop the chunk's own "[" and "]"
        out.append(b"]}")
        super().__init__(content=b"".join(out))


@app.post("/associate_card", status_code=201, summary="Create a new association of a credit card number with a phone number.")
def associate_card(payload: AssociateCardRequest, response: Response):
    _probe_t = aging_probe.t0()  # AGING-PROBE
    with db_conn() as conn:
        _probe_cur = conn.execute(  # AGING-PROBE
            """
            INSERT OR IGNORE INTO associations (credit_card, phone)
            VALUES (?, ?);
            """,
            (payload.credit_card, payload.phone),
        )
    aging_probe.associate_done(_probe_cur.rowcount, _probe_t)  # AGING-PROBE
    response.status_code = status.HTTP_201_CREATED
    # OpenAPI spec only defines the 201 response description (no body)
    return Response(status_code=status.HTTP_201_CREATED)


@app.post(
    "/retrieve_cards",
    response_model=RetrieveCardsResponseSchema,
    summary="Retrieve cards associated with a set of phone numbers. A card must be associated with all given phone numbers to be returned.",
)
def retrieve_cards(payload: RetrieveCardsRequest):
    phones = payload.phone_numbers
    if len(phones) == 1:
        # UNIQUE(credit_card, phone) makes the cards of one phone distinct, so this equals the
        # GROUP BY/HAVING query below; it is answered from idx_assoc_phone_card in order.
        query = "SELECT credit_card FROM associations WHERE phone = ? ORDER BY credit_card ASC;"
        params = (phones[0],)
    else:
        placeholders = ",".join(["?"] * len(phones))

        query = f"""
            SELECT credit_card
            FROM associations
            WHERE phone IN ({placeholders})
            GROUP BY credit_card
            HAVING COUNT(DISTINCT phone) = ?
            ORDER BY credit_card ASC;
        """
        params = (*phones, len(phones))

    _probe_t = aging_probe.t0()  # AGING-PROBE
    with db_conn() as conn:
        rows = conn.execute(query, params).fetchall()
    aging_probe.retrieve_db_done(len(phones), len(rows), _probe_t)  # AGING-PROBE

    if not rows:
        return JSONResponse(status_code=404, content={"detail": "Not found"})

    _probe_resp = RetrieveCardsResponse(card_numbers=[row["credit_card"] for row in rows])  # AGING-PROBE
    aging_probe.retrieve_handler_done(_probe_t)  # AGING-PROBE
    return _probe_resp  # AGING-PROBE


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)