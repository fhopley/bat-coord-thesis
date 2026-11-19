"""model/scripts/export_cap_slide_profiles.py: the import price carries GST
on the spot component when the retail tariff sets gst_on_wholesale_import;
exports and the network column are unchanged."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model.components.billing import GST_RATE
from model.core.prices import build_layer_rates
from model.core.tariff import load_network_tariff, load_retail_tariff
from model.scripts.export_cap_slide_profiles import stack_for_year

TARIFFS = Path(__file__).resolve().parents[1] / 'tariffs'


@pytest.mark.parametrize('retail_id, gst', [('NSW_amber_wholesale_2026', GST_RATE),
                                            ('NSW_operator_wholesale_2026', 0.0)])
def test_import_price_has_gst_on_spot_only(retail_id, gst):
    net = load_network_tariff('NSW_ausgrid_ea025_ea029_2026', TARIFFS)
    retail = load_retail_tariff(retail_id, TARIFFS)
    idx = pd.date_range('2023-01-01 14:05:00', periods=6, freq='5min', tz='Etc/GMT-10')
    prices = pd.Series([100.0, -50.0, 0.0, 15000.0, 300.0, 37.0], index=idx)
    d = stack_for_year(prices, net, retail, 'NSW', 5 / 60)
    lr = build_layer_rates(idx, prices, net, retail, 'NSW', 5 / 60)
    np.testing.assert_allclose(
        d['import_c_kwh'],
        (lr.wholesale_import * (1 + gst) + lr.network_import + lr.retail_import) * 100)
    np.testing.assert_allclose(
        d['export_c_kwh'],
        (lr.wholesale_export + lr.network_export + lr.retail_export) * 100)
    np.testing.assert_allclose(d['network_import_c_kwh'], lr.network_import * 100)
    assert retail.gst_on_wholesale_import is (gst > 0)
