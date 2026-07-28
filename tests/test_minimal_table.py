"""
Smoke test for the minimal epitope table.

Checks a set of well characterised epitopes against their known source proteins and
positions. These are stable facts about the biology rather than about this export, so
a change here means the extraction has drifted, not that IEDB has published more data.
"""

import csv
from pathlib import Path

import polars
import pytest


ROOT = Path(__file__).resolve().parents[1]
MINIMAL_PATH = ROOT / "tmp" / "iedb_parquet" / "iedb_mhc_ligand_minimal.parquet"
OUTPUT_DIR = ROOT / "tmp" / "tests"
OUTPUT_PATH = OUTPUT_DIR / "known_epitopes.csv"

# IEDB already separates residues with a comma inside modified_residues, as in
# "N4,M14", so joining on a comma too flattens the nesting rather than preserving it.
# That is the intent: one separator throughout means a consumer splits once and gets
# every residue, and the grouping into IEDB's own values carries no meaning worth
# keeping. Abbreviations stay pipe separated inside modifications, as IEDB writes them
LIST_SEPARATOR = ","

# sequence -> UniProt accession, start, end
KNOWN_EPITOPES = {
    "GILGFVFTL": ("P03485", 58, 66),  # influenza A M1
    "YLQPRTFLL": ("P0DTC2", 269, 277),  # SARS-CoV-2 spike
    "NLVPMVATV": ("Q6SW59", 495, 503),  # HCMV pp65
    "EVDPIGHLY": ("P43357", 168, 176),  # MAGE-A3
    "SLLQHLIGL": ("P78395", 425, 433),  # PRAME
    "ESDPIVAQY": ("Q8WZ42", 24337, 24345),  # titin
    "VTEHDTLLY": ("F5HC97", 245, 253),  # HCMV DNA polymerase processivity subunit
}

# engineered analogues, so IEDB has no natural source protein for them
ANALOGUE_EPITOPES = ["ELAGIGILTV", "SLLMWITQV"]


@pytest.fixture(scope="module")
def minimal_table():
    """
    Loads the minimal table.

    Returns:
        polars.DataFrame: The minimal table.
    """
    if not MINIMAL_PATH.exists():
        pytest.skip(f"{MINIMAL_PATH} not found, run iedb-minimal first")

    return polars.read_parquet(MINIMAL_PATH)


@pytest.fixture(scope="module")
def known_epitope_csv(minimal_table):
    """
    Writes the best evidenced row for each test epitope to tmp/tests as a CSV.

    List columns are joined on a comma, because CSV has nowhere to put a list. The
    fields that contain one are quoted, so the file still parses as ordinary CSV. This
    is the same set of epitopes the assertions below use, so the file shows what those
    assertions are reading.

    Returns:
        Path: The CSV that was written.
    """
    sequences = list(KNOWN_EPITOPES) + ANALOGUE_EPITOPES

    rows = (
        minimal_table.filter(
            polars.col("sequence").is_in(sequences) & polars.col("allele_is_specific")
        )
        .sort("n_assays", descending=True)
        .unique(subset=["sequence"], keep="first")
        .with_columns(
            polars.col("pmids").cast(polars.List(polars.String)).list.join(LIST_SEPARATOR),
            polars.col("modifications").list.join(LIST_SEPARATOR),
            polars.col("modified_residues").list.join(LIST_SEPARATOR),
        )
        .sort("sequence")
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows.write_csv(OUTPUT_PATH)

    return OUTPUT_PATH


def test_known_epitope_csv_is_written(known_epitope_csv):
    """The CSV lands in tmp/tests with a row for every test epitope."""
    with open(known_epitope_csv, newline="", encoding="utf-8") as output_file:
        rows = list(csv.DictReader(output_file))

    assert len(rows) == len(KNOWN_EPITOPES) + len(ANALOGUE_EPITOPES)
    assert {row["sequence"] for row in rows} == set(KNOWN_EPITOPES) | set(
        ANALOGUE_EPITOPES
    )

    influenza = next(row for row in rows if row["sequence"] == "GILGFVFTL")

    assert influenza["uniprot_id"] == "P03485"
    assert (influenza["start_position"], influenza["end_position"]) == ("58", "66")

    # every joined PMID survives the round trip as a whole number
    pmids = influenza["pmids"].split(LIST_SEPARATOR)

    assert len(pmids) == int(influenza["n_pmids"])
    assert all(pmid.isdigit() for pmid in pmids)


def best_row(minimal_table, sequence):
    """
    Returns the best evidenced specific-allele row for a sequence.

    Args:
        minimal_table (polars.DataFrame): The minimal table.
        sequence (str): The epitope sequence to look up.

    Returns:
        dict: The row with the most assays behind it.
    """
    rows = minimal_table.filter(
        (polars.col("sequence") == sequence) & polars.col("allele_is_specific")
    ).sort("n_assays", descending=True)

    assert rows.height, f"{sequence} is missing from the minimal table"

    return rows.row(0, named=True)


@pytest.mark.parametrize("sequence", KNOWN_EPITOPES)
def test_known_epitopes_keep_their_provenance(minimal_table, sequence):
    """Well characterised epitopes map to the right protein and position."""
    uniprot_id, start, end = KNOWN_EPITOPES[sequence]
    row = best_row(minimal_table, sequence)

    assert row["uniprot_id"] == uniprot_id
    assert (row["start_position"], row["end_position"]) == (start, end)


@pytest.mark.parametrize("sequence", KNOWN_EPITOPES)
def test_positions_span_the_peptide(minimal_table, sequence):
    """
    The reported span is the length of the peptide.

    Aggregating positions across submissions that disagree is the easy way to get this
    wrong: taking the lowest start and highest end reported the titin epitope as 935
    residues long.
    """
    row = best_row(minimal_table, sequence)
    span = row["end_position"] - row["start_position"] + 1

    assert span == row["peptide_length"]


@pytest.mark.parametrize("sequence", ANALOGUE_EPITOPES)
def test_analogues_are_present_without_provenance(minimal_table, sequence):
    """
    Engineered analogues are kept, with null provenance.

    They have no natural source protein, so the nulls are correct rather than missing
    data, but they should still be matchable.
    """
    row = best_row(minimal_table, sequence)

    assert row["uniprot_id"] is None
    assert row["n_pmids"] > 0


def test_sequences_are_residues_only(minimal_table):
    """
    No sequence carries a modification suffix.

    The export appends modifications to the epitope name, as in "LQPFPQPQLPY +
    DEAM(Q8)", which would never match.
    """
    unclean = minimal_table.filter(~polars.col("sequence").str.contains(r"^[A-Z]+$"))

    assert unclean.height == 0


def test_modified_epitopes_are_flagged(minimal_table):
    """A modified entry is distinguishable from an unmodified one."""
    modified = minimal_table.filter(polars.col("is_modified"))

    assert modified.height > 0
    assert modified.get_column("modifications").list.len().min() > 0


def test_grain_is_unique(minimal_table):
    """One row per sequence, epitope ID and restriction."""
    grain = ["sequence", "iedb_epitope_id", "mhc_allele"]

    assert minimal_table.select(grain).n_unique() == minimal_table.height
