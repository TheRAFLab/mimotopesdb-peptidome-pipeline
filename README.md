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

Those composed names are readable but awkward to query, so the Parquet file uses
lower snake case **slugs** instead — `Reference - PMID` becomes `reference_pmid`.
The mapping between the two lives in `iedb/columns/iedb_mhc_ligand_full.json`.

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
uv run iedb-import
```

This writes `tmp/iedb_parquet/iedb_mhc_ligand_full.parquet` and refreshes
`iedb/columns/iedb_mhc_ligand_full.json`, reporting progress as it goes:

```
──────────────────── IEDB peptidome import ─────────────────────
  Found 10 shards, 9.1 GB in tmp/iedb_data
  Wrote 112 column definitions (15 cast) to iedb/columns/iedb_mhc_ligand_full.json

   Rows              5,749,672
   Columns                 112
   Input                9.1 GB
   Output             131.1 MB
   Compression      70x smaller
   Elapsed              21.6 s
```

It then builds the minimal epitope table as a second stage. Paths can be overridden,
and `--skip-minimal` stops after the conversion:

```bash
uv run iedb-import --input /data/iedb --parquet /data/iedb_mhc_ligand_full.parquet
```

See `uv run iedb-import --help` for the full list.

## The minimal epitope table

`tmp/iedb_parquet/iedb_mhc_ligand_minimal.parquet` is a 52.9 MB reduction of the
full export, built for fast sequence matching. It can be rebuilt on its own in about
6 seconds, without redoing the conversion:

```bash
uv run iedb-minimal
```

One row per sequence, IEDB epitope ID and MHC restriction — 3,083,883 rows covering
1,594,878 distinct sequences.

| Column | Notes |
| ------ | ----- |
| `sequence` | residues only, modifications and hybrid-junction spaces stripped |
| `peptide_length` | |
| `iedb_epitope_id` | one sequence can carry several; `GILGFVFTL` has 12 |
| `mhc_allele`, `mhc_class` | |
| `allele_is_specific` | false for class-level restrictions like `HLA class I` |
| `uniprot_id` | 79.6% of rows |
| `source_molecule` | most commonly reported name for the group |
| `start_position`, `end_position` | the modal pair, not the extremes — see below |
| `pmids`, `n_pmids` | every publication behind the group |
| `is_modified`, `modifications`, `modified_residues` | |
| `n_assays` | rows in the export behind this one |

Rows are sorted by sequence, so a lookup skips most row groups on their statistics.
An exact lookup takes around 20 ms and a Levenshtein sweep over all 9-mers around
0.3 s:

```sql
SELECT sequence, mhc_allele, uniprot_id, source_molecule
FROM read_parquet('tmp/iedb_parquet/iedb_mhc_ligand_minimal.parquet')
WHERE peptide_length = 9
  AND damerau_levenshtein(sequence, 'GILGFVFTL') <= 2;
```

### Three things to know about the data

**Positions are modal, not extremes.** Submissions disagree about where an epitope
sits, and taking the lowest start with the highest end describes a peptide that does
not exist: titin's `ESDPIVAQY` comes out as 23410-24345, a 935 residue span for a
nine residue peptide. The most frequently reported start and end are taken together
as a pair, giving 24337-24345. A test asserts every known epitope's span equals its
length.

**Sequences need cleaning before they match.** The export appends modifications to
the epitope name — `LQPFPQPQLPY + DEAM(Q8)` — and writes hybrid peptide fusion
junctions as a space, as in `TEGVEALYLVC KGGS`. Both are stripped out of `sequence`.
The modification is preserved in its own columns with an `is_modified` flag, since
otherwise a modified and an unmodified entry become indistinguishable. The junction is
not: exactly one row of the July 2026 export is gapped, an H2-IAg7-restricted hybrid
insulin peptide, which does not warrant a column across three million rows.

**Engineered analogues have no provenance.** `ELAGIGILTV` (MART-1 A27L) and
`SLLMWITQV` (NY-ESO-1 C165V) are synthetic variants, so IEDB records no source
protein, UniProt ID or position for them. The nulls are correct rather than missing.

Non-peptidic and discontinuous epitopes — 771 rows of glycolipids and small molecules
whose names are chemistry rather than residues — are excluded.

### As a library

`scan_iedb_data` returns a `LazyFrame`, so filters and column selections are pushed
down into the CSV reader and only the data you ask for is read:

```python
import polars

from iedb.read_raw_data import scan_iedb_data

ligands = (
    scan_iedb_data("tmp/iedb_data")
    .filter(polars.col("mhc_restriction_class") == "I")
    .select(["epitope_name", "mhc_restriction_name"])
    .collect()
)
```

Filter before selecting, or the predicate will reference a column that `select` has
already dropped. Pass `slugs=False` to work with the composed IEDB names instead.

### With DuckDB

The slugs need no quoting, so the Parquet file can be queried directly:

```sql
SELECT mhc_restriction_name,
       count(*) AS assays,
       round(avg(epitope_ending_position - epitope_starting_position + 1), 1) AS mean_len
FROM read_parquet('tmp/iedb_parquet/iedb_mhc_ligand_full.parquet')
WHERE mhc_restriction_class = 'I'
  AND reference_date >= 2020
GROUP BY 1
ORDER BY assays DESC
LIMIT 5;
```

```
┌──────────────────────┬─────────┬──────────┐
│ mhc_restriction_name │ assays  │ mean_len │
├──────────────────────┼─────────┼──────────┤
│ HLA class I          │ 1277482 │      9.6 │
│ HLA-A*02:01          │  179540 │      9.6 │
│ HLA-B*07:02          │   52069 │      9.8 │
└──────────────────────┴─────────┴──────────┘
```

Use `import_iedb_data(path, n_rows=10)` to pull a small sample into memory. Note that
`n_rows` applies **per shard**, so `n_rows=10` over 10 shards returns 100 rows; use
`.head()` on the `LazyFrame` if you want a global cap.

## Output

| Path | Contents |
| ---- | -------- |
| `tmp/iedb_parquet/iedb_mhc_ligand_full.parquet` | 5,749,672 rows x 112 columns |
| `tmp/iedb_parquet/iedb_mhc_ligand_minimal.parquet` | 3,083,883 rows x 16 columns, for matching |
| `iedb/columns/iedb_mhc_ligand_full.json` | Every column's index, group, field, name, slug and dtype |
| `iedb/columns/iedb_mhc_ligand_minimal.json` | Each minimal column's dtype, meaning and source |

Both files under `iedb/columns/` are named after the Parquet file they describe, and
are committed as a reference for picking columns and because they diff readably when
a future IEDB export changes shape. An entry in the full file looks like:

```json
{
  "index": 3,
  "group": "Reference",
  "field": "PMID",
  "name": "Reference - PMID",
  "slug": "reference_pmid",
  "dtype": "Int32"
}
```

`slug` is the column name in the Parquet file; `name` is the composed IEDB name it
came from, and `group` plus `field` are the two header rows that compose it. `field`
alone is not unique — `IEDB IRI`, `Name` and `Starting Position` all recur across
groups — which is why the names are composed in the first place.

The minimal file records what each column means and where it came from, with the
dtype read off the built table rather than declared, so it cannot claim a type the
file does not have:

```json
{
  "index": 6,
  "name": "uniprot_id",
  "dtype": "String",
  "description": "UniProt accession of the source protein, where IEDB records one",
  "source": "epitope_molecule_parent_iri, falling back to epitope_source_molecule_iri"
}
```

### Dtypes

97 columns stay as strings. The 15 listed in `CASTS` are cast to `Int16`, `Int32` or
`Float64` — chosen from the observed ranges of the July 2026 export. The cast is
strict, so a value that does not fit raises rather than silently becoming null; a
failure there means the export has changed and `CASTS` needs revisiting. Pass
`cast=False` to `scan_iedb_data` to inspect the raw strings when that happens.

Two fields are easy to misread:

- `host_age` is **not** numeric. It holds free text such as `37-91 years` and
  `child`, and 87% of its values fail to parse as integers, so it stays a string.
- `reference_date` is a publication **year** (1988–2026), not a full date, so it is
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
uv sync           # installs the dev group, including pytest and duckdb
uv run pytest
```

Both test modules write their results to `tmp/tests/` as CSV, so the output can be
eyeballed as well as asserted on:

| Test module | Checks | Writes |
| ----------- | ------ | ------ |
| `tests/test_smoke.py` | The DuckDB query above against the converted export | `class_i_assays_by_mhc_restriction.csv` |
| `tests/test_minimal_table.py` | Well characterised epitopes keep the right protein and position | `known_epitopes.csv` |

These are end-to-end checks rather than unit tests. The first uses the slugged names
unquoted and does arithmetic on the position columns, which only works if they were
cast to integers rather than left as strings. The second pins epitopes such as
`GILGFVFTL` to P03485 at 58-66, which are facts about the biology rather than about
this export, so a failure means the extraction has drifted rather than that IEDB has
published more data.

The Parquet files are build artifacts, so the tests **skip** rather than fail when
the pipeline has not been run.

## License

Released under the [MIT License](LICENSE).
