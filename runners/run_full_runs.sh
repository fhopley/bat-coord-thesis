#!/bin/bash
# Runs from the project root wherever it is started from.
cd "$(dirname "$0")/.."
source venv/bin/activate
# run the following if creating new .sh files 
#   chmod +x runners/run_full_runs.sh
# to run file use
#   ./runners/run_full_runs.sh

# Runs to implement the nudge required for rolling runs with flat TOU prices

# Runs to adjust how AGL minimum VPP event export price is determined & use of 15 minute window

# Last Tesla run
python3 -u -m model.main model/configs/full_run_rolling_tesla.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_rolling_tesla_$(date +%Y%m%d_%H%M%S).log

# Runs to populate all Tesla Electric runs required (ideally with new rates once launched)
# python3 -u -m model.main model/configs/full_run_solar_only_tesla.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_solar_only_tesla_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_self_consumption_tesla.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_self_consumption_tesla_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_optimised_tesla.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_optimised_tesla_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_rolling_tesla_predispatchonly.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_rolling_tesla_predispatchonly_$(date +%Y%m%d_%H%M%S).log

# Runs to populate Amber with more conservative rolling capability and missing EnergyAustralia 
# python3 -u -m model.main model/configs/complete_interupted_run.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/complete_interupted_run_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_optimised_ea_novpp.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_optimised_ea_novpp_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_rolling_amber_predispatchonly.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_rolling_amber_predispatchonly_$(date +%Y%m%d_%H%M%S).log

# Runs to redo AGL perfect foresight with VPP exposure results with the new parameters learnt & to do the rolling vpp exposure runs - all but final rolling run copied across
# python3 -u -m model.main model/configs/full_run_self_consumption_agl.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_self_consumption_agl_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_optimised_agl.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_optimised_agl_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_rolling_self_consumption_agl.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_rolling_self_consumption_agl_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_rolling_agl.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_rolling_agl_$(date +%Y%m%d_%H%M%S).log

# Runs to check a tolerance fix and complete the interrupted rolling origin run - copied across
# python3 -u -m model.main model/configs/test_failed_run.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/test_failed_run_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/complete_interupted_run.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/complete_interupted_run_$(date +%Y%m%d_%H%M%S).log

# Runs to redo Origin perfect foresight with VPP exposure results with the new parameters learnt from the meeting with Origin & to do the rolling vpp exposure runs - copied across
# python3 -u -m model.main model/configs/full_run_self_consumption_origin_lite.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_self_consumption_origin_lite_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_optimised_origin_lite.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_optimised_origin_lite_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_rolling_self_consumption_origin.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_rolling_self_consumption_origin_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_rolling_origin.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_rolling_origin_$(date +%Y%m%d_%H%M%S).log

# Runs to populate the Origin no_vpp configs for the retailer briefing slides - copied across
# python3 -u -m model.main model/configs/full_run_self_consumption_origin_novpp.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_self_consumption_origin_novpp_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_optimised_origin_novpp.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_optimised_origin_novpp_$(date +%Y%m%d_%H%M%S).log

# Remainder of the GloBird runs, with the new parameters learnt from the meeting with GloBird
# python3 -u -m model.main model/configs/full_run_self_consumption_globird.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_self_consumption_globird_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_optimised_globird.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_optimised_globird_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_rolling_globird.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_rolling_globird_$(date +%Y%m%d_%H%M%S).log

# Runs to populate the GloBird no_vpp configs for the retailer briefing slides, rerun with the new parameters learnt from the meeting with GloBird
# python3 -u -m model.main model/configs/full_run_solar_only_globird.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_solar_only_globird_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_self_consumption_globird_novpp.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_self_consumption_globird_novpp_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_optimised_globird_novpp.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_optimised_globird_novpp_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_rolling_globird_novpp.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_rolling_globird_novpp_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/complete_interupted_run.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/complete_interupted_run_$(date +%Y%m%d_%H%M%S).log

# Runs to use the updated Origin rates for the retailer briefing slides, the final run was interupted and completed with complete_interupted_run.yaml
# python3 -u -m model.main model/configs/full_run_solar_only_origin_starter.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_solar_only_origin_starter_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_self_consumption_origin_lite.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_self_consumption_origin_lite_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_optimised_origin_lite.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_optimised_origin_lite_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_rolling_origin_novpp.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_rolling_origin_lite_novpp_$(date +%Y%m%d_%H%M%S).log

# python3 -u -m model.main model/configs/full_run_rolling_agl_novpp.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_rolling_agl_novpp_$(date +%Y%m%d_%H%M%S).log

# python3 -u -m model.main model/configs/full_run_rolling_ea_novpp.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_rolling_ea_novpp_$(date +%Y%m%d_%H%M%S).log

# python3 -u -m model.main model/configs/full_run_optimised_ea.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_optimised_ea_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_rolling_ea.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_rolling_ea_$(date +%Y%m%d_%H%M%S).log

# python3 -u -m model.main model/configs/full_run_optimised_agl_novpp.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_optimised_agl_novpp_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_self_consumption_agl_novpp.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_self_consumption_agl_novpp_$(date +%Y%m%d_%H%M%S).log

# python3 -u -m model.main model/configs/full_run_optimised_origin_max.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_optimised_origin_max_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_optimised_origin_lite.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_optimised_origin_lite_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_optimised_ea.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_optimised_ea_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_optimised_agl.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_optimised_agl_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_optimised_globird.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_optimised_globird_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_optimised_amber.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_optimised_amber_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_self_consumption_origin_max.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_self_consumption_origin_max_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_self_consumption_origin_lite.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_self_consumption_origin_lite_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_self_consumption_ea.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_self_consumption_ea_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_self_consumption_agl.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_self_consumption_agl_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_self_consumption_globird.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_self_consumption_globird_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_self_consumption_amber.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_self_consumption_amber_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_solar_only_origin_max.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_solar_only_origin_max_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_solar_only_origin_starter.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_solar_only_origin_starter_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_solar_only_ea.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_solar_only_ea_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_solar_only_agl.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_solar_only_agl_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_solar_only_globird.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_solar_only_globird_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_solar_only_amber.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_solar_only_amber_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_no_cer_amber.yaml --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_no_cer_amber_$(date +%Y%m%d_%H%M%S).log

