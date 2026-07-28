"""
Builds a compact epitope table for fast fuzzy matching.

The full export carries 112 columns of assay detail, which is far more than a
sequence match needs to carry in memory. This stage reduces it to the identity and
provenance of each epitope, at one row per sequence, IEDB epitope ID and MHC
restriction.
"""

import argparse
import json
import os
import time

import polars
from rich import box
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table


DEFAULT_PARQUET_PATH = "tmp/iedb_parquet/iedb_mhc_ligand_full.parquet"
DEFAULT_MINIMAL_PATH = "tmp/iedb_parquet/iedb_mhc_ligand_minimal.parquet"
DEFAULT_COLUMN_JSON_PATH = "iedb/columns/iedb_mhc_ligand_minimal.json"

# What each column means and where it came from. The dtypes are not recorded here:
# they are read off the built table instead, so that they cannot drift from what the
# file actually contains. Keys must match the projection in build_minimal_table.
COLUMNS = {
    "sequence": {
        "description": "Epitope residues, upper case, with no modification suffix or hybrid junction",
        "source": "epitope_name",
    },
    "peptide_length": {
        "description": "Number of residues in the sequence",
        "source": "derived from sequence",
    },
    "iedb_epitope_id": {
        "description": "IEDB epitope accession; one sequence can carry several",
        "source": "epitope_epitope_iri",
    },
    "mhc_allele": {
        "description": "MHC restriction as reported, which may be an allele or a class",
        "source": "mhc_restriction_name",
    },
    "mhc_class": {
        "description": "MHC class, I or II",
        "source": "mhc_restriction_class",
    },
    "allele_is_specific": {
        "description": "Whether mhc_allele names an allele rather than a class such as 'HLA class I'",
        "source": "derived from mhc_allele",
    },
    "uniprot_id": {
        "description": "UniProt accession of the source protein, where IEDB records one",
        "source": "epitope_molecule_parent_iri, falling back to epitope_source_molecule_iri",
    },
    "source_molecule": {
        "description": "Name of the originating protein, as most commonly reported for the group",
        "source": "epitope_source_molecule",
    },
    "start_position": {
        "description": "Start of the epitope in the source protein, most commonly reported value",
        "source": "epitope_starting_position",
    },
    "end_position": {
        "description": "End of the epitope in the source protein, paired with start_position",
        "source": "epitope_ending_position",
    },
    "pmids": {
        "description": "PubMed IDs of every publication behind the group",
        "source": "reference_pmid",
    },
    "n_pmids": {
        "description": "Number of distinct publications behind the group",
        "source": "derived from pmids",
    },
    "is_modified": {
        "description": "Whether any assay in the group reported a post-translational modification",
        "source": "derived from modifications",
    },
    "modifications": {
        "description": "Modifications reported for the group, such as Oxidation or Deamidation",
        "source": "epitope_modifications",
    },
    "modified_residues": {
        "description": "Residues carrying those modifications, such as M22",
        "source": "epitope_modified_residues",
    },
    "n_assays": {
        "description": "Number of rows in the full export behind this one",
        "source": "row count",
    },
}

# One row per sequence, epitope and restriction. Anything that varies within a group
# is either reduced to its most common value or collected into a list, so that no row
# describes a combination that never appeared in the export.
GRAIN = ["sequence", "iedb_epitope_id", "mhc_allele"]

# Only these carry a matchable sequence. The rest are glycolipids, small molecules
# and discontinuous epitopes, whose names are chemistry rather than residues.
MATCHABLE_OBJECT_TYPE = "Linear peptide"


def extract_fields(lazy_frame: polars.LazyFrame) -> polars.LazyFrame:
    """
    Selects and renames the fields the minimal table needs.

    Three of these are not simple renames. The sequence has to be split out of
    "Epitope - Name", which appends the modification to the residues for modified
    epitopes, as in "LQPFPQPQLPY + DEAM(Q8)". The epitope ID has to be taken from the
    end of its IRI. The UniProt accession comes from the molecule parent IRI, falling
    back to the source molecule IRI: the source molecule is usually an NCBI protein
    record, so the parent is the better source, covering 88.2% of rows against 68.1%.

    Args:
        lazy_frame (polars.LazyFrame): A query over the converted export.

    Returns:
        polars.LazyFrame: The query reduced to the minimal table's input fields.
    """
    uniprot = polars.coalesce(
        polars.col("epitope_molecule_parent_iri").str.extract(r"uniprot/([A-Z0-9]+)"),
        polars.col("epitope_source_molecule_iri").str.extract(r"uniprot/([A-Z0-9]+)"),
    )

    # a hybrid peptide writes its fusion junction as a space, as in
    # "TEGVEALYLVC KGGS", which has to close up for the residues to match. Exactly one
    # row of the July 2026 export is gapped, so the junction is not worth a column of
    # its own across three million rows
    sequence = (
        polars.col("epitope_name")
        .str.split(" + ")
        .list.first()
        .str.replace_all(r"\s+", "")
    )

    return lazy_frame.filter(
        polars.col("epitope_object_type") == MATCHABLE_OBJECT_TYPE
    ).select(
        sequence.alias("sequence"),
        polars.col("epitope_epitope_iri")
        .str.extract(r"epitope/(\d+)")
        .cast(polars.Int32)
        .alias("iedb_epitope_id"),
        polars.col("mhc_restriction_name").alias("mhc_allele"),
        polars.col("mhc_restriction_class").alias("mhc_class"),
        uniprot.alias("uniprot_id"),
        polars.col("epitope_source_molecule").alias("source_molecule"),
        polars.col("epitope_starting_position").alias("start_position"),
        polars.col("epitope_ending_position").alias("end_position"),
        polars.col("reference_pmid").alias("pmid"),
        polars.col("epitope_modified_residues").alias("modified_residues"),
        polars.col("epitope_modifications").alias("modifications"),
    )


def modal_positions(lazy_frame: polars.LazyFrame) -> polars.LazyFrame:
    """
    Picks the most frequently reported start and end position for each epitope.

    The pair is taken together rather than as two independent modes, which could
    otherwise report a start from one submission alongside an end from another. Taking
    the minimum start and maximum end instead would be worse still: submissions
    disagree, and spanning them describes a peptide that does not exist. Titin's
    ESDPIVAQY, for instance, spans 23410-24345 that way, which is 935 residues for a
    nine residue peptide, against 24337-24345 taken modally.

    Args:
        lazy_frame (polars.LazyFrame): The extracted fields.

    Returns:
        polars.LazyFrame: One start and end position per grain.
    """
    return (
        lazy_frame.group_by(GRAIN + ["start_position", "end_position"])
        .agg(polars.len().alias("n"))
        .sort("n", descending=True)
        .unique(subset=GRAIN, keep="first")
        .select(GRAIN + ["start_position", "end_position"])
    )


def build_minimal_table(parquet_path: str) -> polars.DataFrame:
    """
    Builds the minimal table from the converted export.

    Args:
        parquet_path (str): The path to the converted Parquet file.

    Returns:
        polars.DataFrame: The minimal table, sorted by sequence.
    """
    fields = extract_fields(polars.scan_parquet(parquet_path))

    aggregated = fields.group_by(GRAIN).agg(
        polars.col("mhc_class").drop_nulls().first(),
        polars.col("uniprot_id").drop_nulls().mode().first(),
        polars.col("source_molecule").drop_nulls().mode().first(),
        polars.col("pmid").drop_nulls().unique().sort().alias("pmids"),
        polars.col("modifications").drop_nulls().unique().sort(),
        polars.col("modified_residues").drop_nulls().unique().sort(),
        polars.len().alias("n_assays"),
    )

    return (
        aggregated.join(modal_positions(fields), on=GRAIN, how="left")
        .with_columns(
            polars.col("sequence").str.len_chars().cast(polars.Int16).alias("peptide_length"),
            # a modified and an unmodified entry share a sequence once the modification
            # is split off the name, so the flag is what keeps them apart
            polars.col("modifications").list.len().gt(0).alias("is_modified"),
            # class-level restrictions such as "HLA class I" carry no allele detail
            polars.col("mhc_allele").str.contains(r"[*:]").alias("allele_is_specific"),
            polars.col("pmids").list.len().cast(polars.Int32).alias("n_pmids"),
        )
        .select(
            "sequence",
            "peptide_length",
            "iedb_epitope_id",
            "mhc_allele",
            "mhc_class",
            "allele_is_specific",
            "uniprot_id",
            "source_molecule",
            "start_position",
            "end_position",
            "pmids",
            "n_pmids",
            "is_modified",
            "modifications",
            "modified_residues",
            "n_assays",
        )
        # sorting by sequence lets a lookup skip most row groups on their statistics
        .sort(GRAIN)
        .collect()
    )


def describe_columns(table: polars.DataFrame) -> dict:
    """
    Describes the columns of the minimal table.

    The dtypes are read off the table rather than declared, so the description cannot
    claim a type the file does not have.

    Args:
        table (polars.DataFrame): The built minimal table.

    Returns:
        dict: The column count and an ordered list of column descriptions.

    Raises:
        KeyError: If the table holds a column COLUMNS does not describe.
    """
    columns = []
    for index, (name, dtype) in enumerate(table.schema.items()):
        if name not in COLUMNS:
            raise KeyError(f"{name} is not described in COLUMNS")

        columns.append({"index": index, "name": name, "dtype": str(dtype)} | COLUMNS[name])

    return {"column_count": len(columns), "columns": columns}


def write_column_json(table: polars.DataFrame, output_path: str) -> None:
    """
    Writes the minimal table's column descriptions to an indented JSON file.

    Args:
        table (polars.DataFrame): The built minimal table.
        output_path (str): The path to write the JSON file to.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(describe_columns(table), output_file, indent=2)
        output_file.write("\n")


def write_minimal_table(parquet_path: str, output_path: str) -> polars.DataFrame:
    """
    Builds the minimal table and writes it to Parquet.

    Args:
        parquet_path (str): The path to the converted Parquet file.
        output_path (str): The path to write the minimal table to.

    Returns:
        polars.DataFrame: The minimal table that was written.
    """
    table = build_minimal_table(parquet_path)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    table.write_parquet(output_path, compression="brotli", compression_level=9)

    return table


def run(parquet_path: str, output_path: str, column_json_path: str) -> None:
    """
    Builds the minimal table and reports progress to the terminal.

    Args:
        parquet_path (str): The path to the converted Parquet file.
        output_path (str): The path to write the minimal table to.
        column_json_path (str): The path to write the column JSON file to.
    """
    console = Console()

    console.print()
    console.rule("[bold]Epitope minimal table")

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )

    start = time.perf_counter()
    with progress:
        progress.add_task("  Reducing the export to matchable epitopes ", total=None)
        table = write_minimal_table(parquet_path, output_path)
    elapsed = time.perf_counter() - start

    write_column_json(table, column_json_path)
    console.print(
        f"  Wrote [bold cyan]{table.width}[/] column definitions "
        f"to [dim]{column_json_path}[/]"
    )

    source_bytes = os.path.getsize(parquet_path)
    output_bytes = os.path.getsize(output_path)
    sequences = table.get_column("sequence").n_unique()
    with_uniprot = table.get_column("uniprot_id").is_not_null().sum()

    summary = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    summary.add_column(style="dim")
    summary.add_column(style="bold cyan", justify="right")
    summary.add_row("Rows", f"{table.height:,}")
    summary.add_row("Distinct sequences", f"{sequences:,}")
    summary.add_row("With a UniProt ID", f"{100 * with_uniprot / table.height:.1f}%")
    summary.add_row("Modified", f"{table.get_column('is_modified').sum():,}")
    summary.add_row("Source", f"{source_bytes / 1e6:.1f} MB")
    summary.add_row("Output", f"{output_bytes / 1e6:.1f} MB")
    summary.add_row("Elapsed", f"{elapsed:.1f} s")

    console.print(summary)
    console.print(f"  [green]Done[/] [dim]{output_path}[/]")
    console.print()


def main(argv: list[str] | None = None) -> int:
    """
    Command line entry point, installed as iedb-minimal.

    Args:
        argv (list[str] | None): Arguments to parse, or None to read sys.argv.

    Returns:
        int: The process exit status.
    """
    parser = argparse.ArgumentParser(
        prog="iedb-minimal",
        description="Build the compact epitope table used for fuzzy matching.",
    )
    parser.add_argument(
        "--parquet",
        default=DEFAULT_PARQUET_PATH,
        help=f"converted export to read (default: {DEFAULT_PARQUET_PATH})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_MINIMAL_PATH,
        help=f"minimal table to write (default: {DEFAULT_MINIMAL_PATH})",
    )
    parser.add_argument(
        "--columns",
        default=DEFAULT_COLUMN_JSON_PATH,
        help=f"column JSON file to write (default: {DEFAULT_COLUMN_JSON_PATH})",
    )
    arguments = parser.parse_args(argv)

    if not os.path.exists(arguments.parquet):
        Console(stderr=True).print(
            f"[red]error:[/] {arguments.parquet} not found, run iedb-import first"
        )
        return 1

    run(arguments.parquet, arguments.output, arguments.columns)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
