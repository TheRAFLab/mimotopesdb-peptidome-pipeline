"""
Smoke test for the converted IEDB Parquet file.

Runs a representative DuckDB query against the Parquet output and writes the result
to tmp/tests as a CSV, so that the query's output can be eyeballed as well as
asserted on. The query is deliberately one that exercises the parts of the
conversion that are easy to get wrong: the slugged column names are used unquoted,
and the position columns are used in arithmetic, which only works if they were cast
to integers rather than left as strings.
"""

import csv
from pathlib import Path

import duckdb
import pytest


ROOT = Path(__file__).resolve().parents[1]
PARQUET_PATH = ROOT / "tmp" / "iedb_parquet" / "mhc_ligand_full.parquet"
OUTPUT_DIR = ROOT / "tmp" / "tests"
OUTPUT_PATH = OUTPUT_DIR / "class_i_assays_by_mhc_restriction.csv"

QUERY = """
SELECT mhc_restriction_name,
       count(*) AS assays,
       round(avg(epitope_ending_position - epitope_starting_position + 1), 1) AS mean_len
FROM read_parquet('{parquet_path}')
WHERE mhc_restriction_class = 'I'
  AND reference_date >= 2020
GROUP BY 1
ORDER BY assays DESC
LIMIT 5
"""


@pytest.fixture(scope="module")
def query_result():
    """
    Runs the query and writes the result to CSV.

    The Parquet file is a build artifact rather than a fixture, so the test skips
    instead of failing when the conversion has not been run.

    Returns:
        list[tuple]: The rows returned by the query.
    """
    if not PARQUET_PATH.exists():
        pytest.skip(f"{PARQUET_PATH} not found, run iedb-import first")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    query = QUERY.format(parquet_path=PARQUET_PATH)

    connection = duckdb.connect()
    connection.sql(f"COPY ({query}) TO '{OUTPUT_PATH}' (HEADER, DELIMITER ',')")

    return connection.sql(query).fetchall()


def test_query_returns_rows(query_result):
    """The query returns the five rows it asks for."""
    assert len(query_result) == 5


def test_columns_are_typed(query_result):
    """
    The counts and lengths come back as numbers.

    If the numeric casts had not been applied, the position arithmetic in the query
    would have failed outright, so this also covers the cast.
    """
    for name, assays, mean_length in query_result:
        assert isinstance(name, str) and name
        assert assays > 0
        # MHC class I ligands are short peptides, typically 8-11 residues
        assert 5 < mean_length < 20


def test_rows_are_ordered_by_assay_count(query_result):
    """The rows come back in descending order of assay count."""
    counts = [assays for _, assays, _ in query_result]
    assert counts == sorted(counts, reverse=True)


def test_csv_is_written(query_result):
    """The CSV lands in tmp/tests with a header and one line per row."""
    assert OUTPUT_PATH.exists()

    with open(OUTPUT_PATH, newline="", encoding="utf-8") as output_file:
        rows = list(csv.reader(output_file))

    assert rows[0] == ["mhc_restriction_name", "assays", "mean_len"]
    assert len(rows) == len(query_result) + 1
    assert rows[1][0] == query_result[0][0]
