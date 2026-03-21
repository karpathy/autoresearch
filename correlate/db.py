"""
Postgres data access layer for RiskWise indices.

Handles discovery of available indices/sub-indices and extraction of time series data.
The schema is intentionally flexible — the agent can override discovery queries
to match whatever schema the target database uses.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import pandas as pd
import psycopg2
import psycopg2.extras

from .config import AnalysisConfig

logger = logging.getLogger(__name__)


@dataclass
class IndexMetadata:
    """Metadata for a single risk index."""
    name: str
    category: str | None = None
    description: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    n_observations: int = 0


class RiskWiseDB:
    """Postgres interface for RiskWise index data."""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self._conn = None

    @property
    def conn(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.config.db.connection_string)
        return self._conn

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()

    # ------------------------------------------------------------------
    # Schema discovery
    # ------------------------------------------------------------------

    def discover_tables(self) -> list[str]:
        """Find all tables/views in the configured schema."""
        query = """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = %s
            ORDER BY table_name
        """
        with self.conn.cursor() as cur:
            cur.execute(query, (self.config.db.schema,))
            return [row[0] for row in cur.fetchall()]

    def discover_columns(self, table: str) -> list[dict]:
        """Return column names and types for a table."""
        query = """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position
        """
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, (self.config.db.schema, table))
            return [dict(row) for row in cur.fetchall()]

    # ------------------------------------------------------------------
    # Index discovery
    # ------------------------------------------------------------------

    def discover_indices(self) -> list[IndexMetadata]:
        """
        Discover all available risk indices in the database.

        This uses a default query that assumes a long-format table with columns:
            (date, index_name, value, [category], [description])

        Override `_build_discovery_query` if your schema differs.
        """
        query = self._build_discovery_query()
        df = pd.read_sql(query, self.conn)
        indices = []
        for _, row in df.iterrows():
            indices.append(IndexMetadata(
                name=row["index_name"],
                category=row.get("category"),
                description=row.get("description"),
                start_date=row.get("start_date"),
                end_date=row.get("end_date"),
                n_observations=int(row.get("n_observations", 0)),
            ))
        logger.info("Discovered %d indices", len(indices))
        return indices

    def _build_discovery_query(self) -> str:
        """
        Build SQL to discover available indices. Override this if your schema
        is different from the default long-format table.
        """
        table = self.config.index_table_pattern
        name_col = self.config.index_name_column
        date_col = self.config.index_date_column

        return f"""
            SELECT
                {name_col} AS index_name,
                MIN({date_col}) AS start_date,
                MAX({date_col}) AS end_date,
                COUNT(*) AS n_observations
            FROM {self.config.db.schema}.{table}
            GROUP BY {name_col}
            ORDER BY {name_col}
        """

    # ------------------------------------------------------------------
    # Time series extraction
    # ------------------------------------------------------------------

    def get_index_series(
        self,
        index_name: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """
        Fetch a single index as a time series DataFrame with columns [date, value].
        Sorted by date ascending. Suitable for merging with market data.
        """
        table = self.config.index_table_pattern
        name_col = self.config.index_name_column
        date_col = self.config.index_date_column
        val_col = self.config.index_value_column

        query = f"""
            SELECT {date_col} AS date, {val_col} AS value
            FROM {self.config.db.schema}.{table}
            WHERE {name_col} = %s
        """
        params: list = [index_name]

        if start_date:
            query += f" AND {date_col} >= %s"
            params.append(start_date)
        if end_date:
            query += f" AND {date_col} <= %s"
            params.append(end_date)

        query += f" ORDER BY {date_col} ASC"

        df = pd.read_sql(query, self.conn, params=params)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        return df

    def get_all_indices_wide(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """
        Fetch ALL indices as a wide-format DataFrame: rows = dates, columns = index names.
        Missing values are left as NaN (indices may have different date ranges).
        """
        table = self.config.index_table_pattern
        name_col = self.config.index_name_column
        date_col = self.config.index_date_column
        val_col = self.config.index_value_column

        query = f"""
            SELECT {date_col} AS date, {name_col} AS index_name, {val_col} AS value
            FROM {self.config.db.schema}.{table}
        """
        conditions = []
        params: list = []

        if start_date:
            conditions.append(f"{date_col} >= %s")
            params.append(start_date)
        if end_date:
            conditions.append(f"{date_col} <= %s")
            params.append(end_date)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += f" ORDER BY {date_col}"

        df = pd.read_sql(query, self.conn, params=params)
        if df.empty:
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"])
        wide = df.pivot_table(index="date", columns="index_name", values="value")
        wide = wide.sort_index()
        return wide

    # ------------------------------------------------------------------
    # Custom SQL escape hatch
    # ------------------------------------------------------------------

    def run_query(self, sql: str, params: list | None = None) -> pd.DataFrame:
        """Run an arbitrary read-only query and return a DataFrame."""
        return pd.read_sql(sql, self.conn, params=params)
