"""Sites excluded from every analysis, and why.

Single source of truth. Previously each of financial_analysis.py,
flow_exposure_analysis.py and cap_account_analysis.py held its own copy of
this set, and they drifted: financial_analysis and flow_exposure carried
{'S0588'} while cap_account carried {'S0588', 'W0164', 'S0241'}. W0164 was
therefore present in the financial and flow-exposure results and absent
from the cap account results, unnoticed because nothing compared them.

Kept dependency-free on purpose. financial_analysis.py would be the natural
home, but it sets global matplotlib rcParams at import time and pulls in
matplotlib, scipy and yaml, so importing it from a script that only needs
this constant would restyle that script's charts as a side effect.

THE CRITERION: annual gross household load too large to be residential,
from `avg_daily_gross_load_kwh` in the EDP site metadata
(data/processed/edp/{year}_selected_sites_clean.csv).

Measured across NSW/QLD/SA/VIC 2021-2025, the retained population has a
median annual load of 7,200 kWh and a maximum of 26,300. Against that:

    S0588    89,800 - 107,700 kWh/yr    12-15x median
    W0164   107,400 - 114,300 kWh/yr    15-16x median

At the top that is about 310 kWh a day, which is not a dwelling. S0588 is
also the ~$27k-baseline site that dominated early "biggest winner"
findings in the cap account analysis.

NOT excluded, though previously listed:

    S0241    27,300 kWh/yr

S0241 was excluded with no recorded reason and does not meet this test: it
sits inside the range of retained sites and below three of them. Retained.

Also deliberately retained are the six high-load sites that benefit from
the cap account's import forgiveness -- S0176, S0172, S0155, S0579, S0308,
W0179, spanning 10,300-30,300 kWh/yr. They sit on a continuous tail of the
population rather than detached from it, their PV is unremarkable, and they
are the only sites in 1,428 site-years that are net above-cap importers.
Removing them would leave the import side of the cap offers with no
beneficiaries at all. See cap_account_spec.md section 3.

Add a site here only against the stated criterion, and record the figure
that put it over the line, so "what else was removed?" has an answer.
"""
from __future__ import annotations

EXCLUDED_SITES: frozenset[str] = frozenset({'S0588', 'W0164'})
