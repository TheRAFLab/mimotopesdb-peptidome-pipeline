# mimotopesdb-peptidome-pipeline

> Converts the sharded IEDB MHC ligand export into a single typed Parquet file.

## Overview

The IEDB `mhc_ligand_full` export arrives as a set of CSV shards totalling around
9 GB. This pipeline reads them, gives them consistent column names and dtypes, and
streams the result into one Parquet file that later steps can query cheaply.

Two quirks of the export drive most of the design:

- **The header is two rows.** Row 0 is a category group (`Reference`, `Epitope`) and
  row 1 is a field name (`PMID`, `IEDB IRI`). Neither is unique on its own, so the
  two are joined into names like `Reference - PMID`.
- **Only the first shard carries that header.** Shards 01 onwards begin directly with
  data. Reading them with headers enabled makes Polars treat the first record of each
  shard as a header row, which both loses that record and makes the shards refuse to
  concatenate.

Every shard is therefore read with headers disabled and given the names taken from
shard 00. Columns are read as strings so that per-shard dtype inference cannot make
two shards disagree, then the genuinely numeric columns are cast once at the end.

## Requirements

- Python >= 3.14
- [uv](https://docs.astral.sh/uv/) for dependency management

## Installation

```bash
git clone https://github.com/TheRAFLab/mimotopesdb-peptidome-pipeline.git
cd mimotopesdb-peptidome-pipeline
uv sync
```

## Input

Place the IEDB shards in `tmp/iedb_data/`, named as IEDB ships them:

```
tmp/iedb_data/
├── mhc_ligand_full_00.csv    # the only shard with the header
├── mhc_ligand_full_01.csv
└── ...
```

`tmp/` is gitignored, so the raw data and the Parquet output stay out of version
control.

## Usage

```bash
uv run iedb/read_raw_data.py
```

This writes `tmp/iedb_parquet/mhc_ligand_full.parquet` and refreshes
`iedb/columns.json`, reporting progress as it goes:

```
──────────────────── IEDB peptidome import ─────────────────────
  Found 10 shards, 9.1 GB in tmp/iedb_data
  Wrote 112 column definitions (15 cast) to iedb/columns.json

   Rows              5,749,672
   Columns                 112
   Input                9.1 GB
   Output             131.1 MB
   Compression      70x smaller
   Elapsed              23.6 s
```

### As a library

`scan_iedb_data` returns a `LazyFrame`, so filters and column selections are pushed
down into the CSV reader and only the data you ask for is read:

```python
import polars

from iedb.read_raw_data import scan_iedb_data

ligands = (
    scan_iedb_data("tmp/iedb_data")
    .filter(polars.col("MHC Restriction - Class") == "I")
    .select(["Epitope - Name", "MHC Restriction - Name"])
    .collect()
)
```

Filter before selecting, or the predicate will reference a column that `select` has
already dropped.

Use `import_iedb_data(path, n_rows=10)` to pull a small sample into memory. Note that
`n_rows` applies **per shard**, so `n_rows=10` over 10 shards returns 100 rows; use
`.head()` on the `LazyFrame` if you want a global cap.

## Output

| Path | Contents |
| ---- | -------- |
| `tmp/iedb_parquet/mhc_ligand_full.parquet` | 5,749,672 rows x 112 columns |
| `iedb/columns.json` | Every column's index, group, field, name and dtype |

`columns.json` is committed as a reference for picking columns, and because it diffs
readably when a future IEDB export changes shape.

### Dtypes

97 columns stay as strings. The 15 listed in `CASTS` are cast to `Int16`, `Int32` or
`Float64` — chosen from the observed ranges of the July 2026 export. The cast is
strict, so a value that does not fit raises rather than silently becoming null; a
failure there means the export has changed and `CASTS` needs revisiting. Pass
`cast=False` to `scan_iedb_data` to inspect the raw strings when that happens.

Two fields are easy to misread:

- `Host - Age` is **not** numeric. It holds free text such as `37-91 years` and
  `child`, and 87% of its values fail to parse as integers, so it stays a string.
- `Reference - Date` is a publication **year** (1988–2026), not a full date, so it is
  an `Int16`.

### Compression

Parquet is written with brotli at level 9. The level is set explicitly because
brotli's Parquet default is low enough to be counterproductive. Measured over the
July 2026 export:

| Codec | Size | Time |
| ----- | ---- | ---- |
| zstd, default level | 141.0 MB | 17.5 s |
| brotli, default level | 155.1 MB | 17.9 s |
| zstd, level 9 | 132.1 MB | 17.1 s |
| **brotli, level 9** | **131.1 MB** | **21.2 s** |

Timings are warm-cache and dominated by CSV parsing rather than compression. zstd at
level 9 is within 1% on size and noticeably quicker to write, so it is the better
choice if this conversion ever runs frequently.

## Development

```bash
uv sync --all-extras   # install dev dependencies
uv run pytest          # run the test suite
```

## License

Released under the [MIT License](LICENSE).
