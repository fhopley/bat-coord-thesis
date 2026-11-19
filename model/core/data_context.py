"""DataContext: the read-only input bundle passed to model components.

Produced once by ingest.py for a single scenario, then passed to downstream
components alongside any intermediate outputs (SiteTraces, VppTraces).

DataContext is intentionally read-only by convention. Components must not
mutate its contents - neither the top-level dataclass fields (enforced by
frozen=True) nor the underlying DataFrames and Series (enforced only by
convention, since pandas objects are mutable).

If a component needs to modify any input, it must take a defensive copy
(df.copy()) at the top of the function.
"""
from dataclasses import dataclass
from typing import Mapping
import pandas as pd

from .scenario import ScenarioConfig
from .tariff import NetworkTariff, RetailTariff


@dataclass(frozen=True)
class DataContext:
    """Input data bundle for a single scenario.

    Attributes:
        site_traces: Per-site 5-minute energy traces, keyed by site_id.
            Each DataFrame contains the cleaned and infilled EDP data
            (gross load, PV, flags, infill markers) for the scenario
            year. Index is datetime_aest (5-min interval-end). Energy
            columns in kWh per interval.
        prices: 5-min NEM Regional Reference Price for the scenario's
            state and year, in $/MWh. Index is datetime_aest.
        site_metadata: Site-level metadata (state, inverter size, etc.)
            filtered to the sites included in site_traces.
        network_tariff: Network charges and loss factors for the scenario.
            Always present; used for the network layer and (via its loss
            factors) the wholesale layer.
        retail_tariff: The retail plan the sites are billed under (tou,
            wholesale_exposed or mixed).
        scenario: The ScenarioConfig that produced this context.
    """
    site_traces: Mapping[str, pd.DataFrame]
    prices: pd.Series
    site_metadata: pd.DataFrame
    network_tariff: NetworkTariff
    retail_tariff: RetailTariff
    scenario: ScenarioConfig