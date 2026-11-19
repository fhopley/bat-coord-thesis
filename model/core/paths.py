"""Filesystem path configuration for the model.

A Paths instance holds the locations of the input data and tariff files.
Passing a Paths object to ingest() rather than hardcoding paths keeps
ingest pure and makes testing straightforward — tests construct a Paths
pointing at fixture directories.

The default Paths assumes the model is invoked from the project root
(matching the convention in the rest of the project).
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    """Locations of input data and configuration."""
    edp_clean_dir: Path                # contains YYYY/edp_clean_{site}_{YYYY}.parquet
    edp_metadata_dir: Path             # contains YYYY_selected_sites_clean.csv
    nem_prices_dir: Path               # contains YYYY_nem_prices.parquet files
    tariffs_dir: Path                  # contains tariff YAML files


def default_paths(project_root: Path | None = None) -> Paths:
    """Return Paths anchored at the project root.

    If project_root is not provided, uses the current working directory
    (matching the "always run from project root" convention).
    """
    root = project_root if project_root is not None else Path.cwd()
    return Paths(
        edp_clean_dir=root / 'data' / 'clean' / 'edp',
        edp_metadata_dir=root / 'data' / 'processed' / 'edp',
        nem_prices_dir=root / 'data' / 'clean' / 'nem_prices',
        tariffs_dir=root / 'model' / 'tariffs',
    )
