import argparse
import json
import os
import re
import time

import polars
from rich import box
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table


# The IEDB export has a two-row header: row 0 is the category group ("Reference",
# "Epitope", ...) and row 1 is the field name ("PMID", "IEDB IRI", ...). Neither row
# is unique on its own, so the two are joined to form the column names.
HEADER_ROWS = 2

# Only this shard carries the header; the rest are bare continuation rows.
HEADER_FILE = "mhc_ligand_full_00.csv"

# Defaults for the command line entry point. The data paths sit under tmp/, which is
# gitignored, because the export and its Parquet run to gigabytes.
DEFAULT_INPUT_PATH = "tmp/iedb_data"
DEFAULT_PARQUET_PATH = "tmp/iedb_parquet/mhc_ligand_full.parquet"
DEFAULT_COLUMN_JSON_PATH = "iedb/columns.json"

# Every shard is read as strings so that the shards concatenate cleanly, then the
# columns that are genuinely numeric are cast back once, here. Widths are chosen
# from the observed ranges of the July 2026 export, with headroom. Note that
# "Host - Age" is deliberately absent: it is free text ("37-91 years", "child").
CASTS = {
    "Reference - PMID": polars.Int32,
    "Reference - Submission ID": polars.Int32,
    # a publication year, not a full date, despite the field name
    "Reference - Date": polars.Int16,
    "Epitope - Starting Position": polars.Int32,
    "Epitope - Ending Position": polars.Int32,
    "Related Object - Starting Position": polars.Int32,
    "Related Object - Ending Position": polars.Int32,
    "in vivo Antigen - Starting Position": polars.Int32,
    "in vivo Antigen - Ending Position": polars.Int32,
    "In vitro Process - Starting Position": polars.Int32,
    "In vitro Process - Ending Position": polars.Int32,
    "Assay - Number of Subjects Tested": polars.Int32,
    "Assay - Number of Subjects Responded": polars.Int32,
    "Assay - Quantitative measurement": polars.Float64,
    "Assay - Response Frequency (%)": polars.Float64,
}


def slugify(name: str) -> str:
    """
    Converts an IEDB column name into a slug that SQL can use unquoted.

    The composed IEDB names contain spaces, hyphens, parentheses and a percent sign,
    so they have to be quoted in every query that touches them. The slug is lower
    snake case, which DuckDB and friends accept bare.

    Args:
        name (str): The composed column name, e.g. "Assay - Response Frequency (%)".

    Returns:
        str: The slug, e.g. "assay_response_frequency_percent".
    """
    # spelled out, so that "Response Frequency (%)" keeps its meaning
    text = name.replace("%", "percent")

    return re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_").lower()


def slug_columns(lazy_frame: polars.LazyFrame) -> polars.LazyFrame:
    """
    Renames the columns of a lazy query to their slugs.

    Args:
        lazy_frame (polars.LazyFrame): The lazy query to rename.

    Returns:
        polars.LazyFrame: The query with slugged column names.

    Raises:
        ValueError: If two column names slug to the same string.
    """
    names = lazy_frame.collect_schema().names()
    slugs = [slugify(name) for name in names]

    if len(set(slugs)) != len(slugs):
        duplicates = {slug for slug in slugs if slugs.count(slug) > 1}
        raise ValueError(f"column names are not unique once slugged: {duplicates}")

    return lazy_frame.rename(dict(zip(names, slugs)))


def list_shards(file_path: str) -> list[str]:
    """
    Lists the IEDB CSV shards in shard order.

    Args:
        file_path (str): The path to the directory of CSV files.

    Returns:
        list[str]: The shard file names, with the shard carrying the header first.

    Raises:
        FileNotFoundError: If the directory holds no shards, or the shard carrying
            the header is missing.
    """
    files = [f for f in os.listdir(file_path) if f.endswith(".csv")]

    # sort the files so that the file with the header is first
    files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    if not files:
        raise FileNotFoundError(f"no CSV shards found in {file_path}")

    if files[0] != HEADER_FILE:
        raise FileNotFoundError(f"{HEADER_FILE} is missing from {file_path}")

    return files


def read_header_rows(file_path: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """
    Reads the two rows of the IEDB header separately.

    Args:
        file_path (str): The path to the CSV file containing the header.

    Returns:
        tuple[tuple[str, ...], tuple[str, ...]]: The category groups and the field
            names, in column order.
    """
    header = polars.read_csv(
        file_path,
        has_header=False,
        n_rows=HEADER_ROWS,
        infer_schema=False,
    )
    return header.row(0), header.row(1)


def read_header(file_path: str) -> list[str]:
    """
    Reads the two-row IEDB header and flattens it into unique column names.

    Args:
        file_path (str): The path to the CSV file containing the header.

    Returns:
        list[str]: The flattened column names, e.g. "Reference - PMID".
    """
    groups, fields = read_header_rows(file_path)
    return [f"{group} - {field}" for group, field in zip(groups, fields)]


def describe_columns(file_path: str) -> dict:
    """
    Describes the columns of the IEDB export.

    Column order is significant, because the shards are read positionally, so the
    columns are described as an ordered list rather than as a mapping.

    Args:
        file_path (str): The path to the directory of CSV files.

    Returns:
        dict: The column count and an ordered list of column descriptions.
    """
    groups, fields = read_header_rows(os.path.join(file_path, HEADER_FILE))

    columns = []
    for index, (group, field) in enumerate(zip(groups, fields)):
        name = f"{group} - {field}"
        columns.append(
            {
                "index": index,
                "group": group,
                "field": field,
                "name": name,
                "slug": slugify(name),
                "dtype": str(CASTS.get(name, polars.String)),
            }
        )

    return {"column_count": len(columns), "columns": columns}


def write_column_json(file_path: str, output_path: str) -> None:
    """
    Writes the column descriptions to an indented JSON file.

    This is a reference document for anyone working out which of the 112 columns they
    need, and it diffs cleanly, so a change to a future IEDB export shows up as a
    readable change here.

    Args:
        file_path (str): The path to the directory of CSV files.
        output_path (str): The path to write the JSON file to.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(describe_columns(file_path), output_file, indent=2)
        output_file.write("\n")


def scan_raw_data(
    file_path: str,
    columns: list[str],
    skip_rows: int = 0,
    n_rows: int | None = None,
) -> polars.LazyFrame:
    """
    Lazily scans one raw IEDB shard.

    Every shard is scanned without a header and given the same column names, so that
    the shards can be concatenated. All columns are read as strings to stop Polars
    inferring a different dtype per shard, which would break the concatenation just
    as mismatched names would.

    Args:
        file_path (str): The path to the CSV file.
        columns (list[str]): The column names to apply to the shard.
        skip_rows (int): Leading rows to discard, used to skip the header rows.
        n_rows (int | None): Row limit, for sampling during development.

    Returns:
        polars.LazyFrame: A lazy query over the shard. No data is read yet.
    """
    return polars.scan_csv(
        file_path,
        has_header=False,
        skip_rows=skip_rows,
        new_columns=columns,
        infer_schema=False,
        n_rows=n_rows,
    )


def cast_columns(lazy_frame: polars.LazyFrame) -> polars.LazyFrame:
    """
    Casts the numeric IEDB columns from string to their proper dtypes.

    The cast is strict, so a value that does not fit raises rather than silently
    becoming null. Every value in the July 2026 export converts cleanly, so a failure
    here means the export has changed shape and CASTS needs revisiting.

    Args:
        lazy_frame (polars.LazyFrame): The lazy query to cast.

    Returns:
        polars.LazyFrame: The query with the numeric columns cast.
    """
    return lazy_frame.with_columns(
        polars.col(column).cast(dtype) for column, dtype in CASTS.items()
    )


def scan_iedb_data(
    file_path: str,
    n_rows: int | None = None,
    cast: bool = True,
    slugs: bool = True,
) -> polars.LazyFrame:
    """
    Builds a lazy query over the full set of sharded IEDB CSV files.

    Nothing is read into memory here beyond the header. Chain filters and column
    selections onto the result before collecting or sinking it, so that Polars can
    push them down into the CSV reader and skip the data it does not need.

    Args:
        file_path (str): The path to the directory of CSV files.
        n_rows (int | None): Row limit per shard, for sampling during development.
        cast (bool): Whether to cast the numeric columns. Set to False to inspect
            the raw strings when a cast fails.
        slugs (bool): Whether to rename the columns to their slugs. Set to False to
            keep the composed IEDB names, which match the export's own header.

    Returns:
        polars.LazyFrame: A lazy query over every shard, concatenated in shard order.
    """
    files = list_shards(file_path)
    columns = read_header(os.path.join(file_path, HEADER_FILE))

    lazy_frames = [
        scan_raw_data(
            os.path.join(file_path, file),
            columns,
            skip_rows=HEADER_ROWS if file == HEADER_FILE else 0,
            n_rows=n_rows,
        )
        for file in files
    ]

    combined = polars.concat(lazy_frames)

    # cast before renaming, because CASTS is keyed by the composed IEDB names
    if cast:
        combined = cast_columns(combined)

    return slug_columns(combined) if slugs else combined


def import_iedb_data(file_path: str, n_rows: int | None = None) -> polars.DataFrame:
    """
    Imports IEDB data from a set of sharded CSV files into memory.

    The full dataset is far larger than the CSVs on disk once loaded, so this is
    intended for sampling with n_rows. To process everything, use convert_to_parquet
    or chain onto scan_iedb_data instead.

    Args:
        file_path (str): The path to the directory of CSV files.
        n_rows (int | None): Row limit per shard, for sampling during development.

    Returns:
        polars.DataFrame: A Polars DataFrame containing the IEDB data.
    """
    return scan_iedb_data(file_path, n_rows=n_rows).collect()


def convert_to_parquet(file_path: str, output_path: str) -> None:
    """
    Streams the sharded IEDB CSV files into a single Parquet file.

    The streaming engine processes the data in batches, so peak memory stays bounded
    regardless of how large the export is. The result is a columnar file that later
    steps can scan selectively rather than re-parsing gigabytes of CSV.

    Brotli is used for compression, at an explicit level because its Parquet default
    is low. Measured over the July 2026 export: brotli at the default level produces
    155 MB in 17.9s, at level 9 it produces 131 MB in 21.2s. The extra few seconds
    are worth it for data that is written once and read often.

    Args:
        file_path (str): The path to the directory of CSV files.
        output_path (str): The path to write the Parquet file to.
    """
    scan_iedb_data(file_path).sink_parquet(
        output_path,
        compression="brotli",
        compression_level=9,
        engine="streaming",
        mkdir=True,
    )


def run(file_path: str, parquet_path: str, column_json_path: str) -> None:
    """
    Runs the import and reports progress to the terminal.

    The reporting lives here rather than in the functions above, so that they stay
    usable as a library. Polars exposes no progress callback for a streaming sink, so
    the conversion is shown as an elapsed-time spinner rather than a percentage: the
    bar would have to invent a position it cannot know.

    Args:
        file_path (str): The path to the directory of CSV files.
        parquet_path (str): The path to write the Parquet file to.
        column_json_path (str): The path to write the column JSON file to.
    """
    console = Console()

    shards = list_shards(file_path)
    input_bytes = sum(os.path.getsize(os.path.join(file_path, s)) for s in shards)

    console.print()
    console.rule("[bold]IEDB peptidome import")
    console.print(
        f"  Found [bold cyan]{len(shards)}[/] shards, "
        f"[bold cyan]{input_bytes / 1e9:.1f} GB[/] in [dim]{file_path}[/]"
    )

    columns = describe_columns(file_path)
    write_column_json(file_path, column_json_path)
    cast_count = sum(1 for column in columns["columns"] if column["dtype"] != "String")
    console.print(
        f"  Wrote [bold cyan]{columns['column_count']}[/] column definitions "
        f"([bold cyan]{cast_count}[/] cast) to [dim]{column_json_path}[/]"
    )

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )

    start = time.perf_counter()
    with progress:
        progress.add_task("  Streaming shards to Parquet (compressed with brotli at level 9) ", total=None)
        convert_to_parquet(file_path, parquet_path)
    elapsed = time.perf_counter() - start

    output_bytes = os.path.getsize(parquet_path)
    rows = polars.scan_parquet(parquet_path).select(polars.len()).collect().item()

    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column(style="bold cyan", justify="right")
    table.add_row("Rows", f"{rows:,}")
    table.add_row("Columns", f"{columns['column_count']:,}")
    table.add_row("Input", f"{input_bytes / 1e9:.1f} GB")
    table.add_row("Output", f"{output_bytes / 1e6:.1f} MB")
    table.add_row("Compression", f"{input_bytes / output_bytes:.0f}x smaller")
    table.add_row("Elapsed", f"{elapsed:.1f} s")

    console.print(table)
    console.print(f"  [green]Done[/] [dim]{parquet_path}[/]")
    console.print()


def main(argv: list[str] | None = None) -> int:
    """
    Command line entry point, installed as iedb-import.

    Args:
        argv (list[str] | None): Arguments to parse, or None to read sys.argv.

    Returns:
        int: The process exit status.
    """
    parser = argparse.ArgumentParser(
        prog="iedb-import",
        description="Convert the sharded IEDB MHC ligand export to Parquet.",
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_PATH,
        help=f"directory holding the CSV shards (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--parquet",
        default=DEFAULT_PARQUET_PATH,
        help=f"Parquet file to write (default: {DEFAULT_PARQUET_PATH})",
    )
    parser.add_argument(
        "--columns",
        default=DEFAULT_COLUMN_JSON_PATH,
        help=f"column JSON file to write (default: {DEFAULT_COLUMN_JSON_PATH})",
    )
    arguments = parser.parse_args(argv)

    try:
        run(arguments.input, arguments.parquet, arguments.columns)
    except FileNotFoundError as error:
        # the usual cause is the export not having been downloaded yet, which is a
        # user error rather than a crash, so report it without a traceback
        Console(stderr=True).print(f"[red]error:[/] {error}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
