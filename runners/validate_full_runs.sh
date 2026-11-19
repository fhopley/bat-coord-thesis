#!/bin/bash
# Runs from the project root wherever it is started from.
cd "$(dirname "$0")/.."
source venv/bin/activate
# run the following if creating new .sh files 
#   chmod +x runners/validate_full_runs.sh
# to run file use
#   ./runners/validate_full_runs.sh

python3 -u -m model.main model/configs/full_run_rolling_tesla.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_rolling_tesla_$(date +%Y%m%d_%H%M%S).log

# python3 -u -m model.main model/configs/full_run_solar_only_tesla.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_solar_only_tesla_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_self_consumption_tesla.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_self_consumption_tesla_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_optimised_tesla.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_optimised_tesla_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_rolling_tesla_predispatchonly.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_rolling_tesla_predispatchonly_$(date +%Y%m%d_%H%M%S).log

# python3 -u -m model.main model/configs/complete_interupted_run.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/complete_interupted_run_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_optimised_ea_novpp.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_optimised_ea_novpp_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_rolling_amber_predispatchonly.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_rolling_amber_predispatchonly_$(date +%Y%m%d_%H%M%S).log

# python3 -u -m model.main model/configs/full_run_self_consumption_agl.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_self_consumption_agl_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_optimised_agl.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_optimised_agl_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_rolling_self_consumption_agl.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_rolling_self_consumption_agl_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_rolling_agl.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_rolling_agl_$(date +%Y%m%d_%H%M%S).log

# python3 -u -m model.main model/configs/test_failed_run.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/test_failed_run_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/complete_interupted_run.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/complete_interupted_run_$(date +%Y%m%d_%H%M%S).log

# python3 -u -m model.main model/configs/full_run_self_consumption_origin_lite.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_self_consumption_origin_lite_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_optimised_origin_lite.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_optimised_origin_lite_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_rolling_self_consumption_origin.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_rolling_self_consumption_origin_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_rolling_origin.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_rolling_origin_$(date +%Y%m%d_%H%M%S).log

# python3 -u -m model.main model/configs/full_run_self_consumption_origin_novpp.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_self_consumption_origin_novpp_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_optimised_origin_novpp.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_optimised_origin_novpp_$(date +%Y%m%d_%H%M%S).log

# python3 -u -m model.main model/configs/full_run_self_consumption_globird.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_self_consumption_globird_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_optimised_globird.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_optimised_globird_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_rolling_globird.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_rolling_globird_$(date +%Y%m%d_%H%M%S).log

# python3 -u -m model.main model/configs/full_run_solar_only_globird.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_solar_only_globird_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_self_consumption_globird_novpp.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_self_consumption_globird_novpp_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_optimised_globird_novpp.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_optimised_globird_novpp_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_rolling_globird_novpp.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_rolling_globird_novpp_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/complete_interupted_run.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/complete_interupted_run_$(date +%Y%m%d_%H%M%S).log

# python3 -u -m model.main model/configs/full_run_solar_only_origin_starter.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_solar_only_origin_starter_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_self_consumption_origin_lite.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_self_consumption_origin_lite_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_optimised_origin_lite.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_optimised_origin_lite_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_rolling_origin_novpp.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_rolling_origin_lite_novpp_$(date +%Y%m%d_%H%M%S).log

# python3 -u -m model.main model/configs/full_run_rolling_agl_novpp.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_rolling_agl_novpp_$(date +%Y%m%d_%H%M%S).log

# python3 -u -m model.main model/configs/full_run_rolling_ea_novpp.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_rolling_ea_novpp_$(date +%Y%m%d_%H%M%S).log

# python3 -u -m model.main model/configs/full_run_optimised_ea.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_optimised_ea_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_rolling_ea.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_rolling_ea_$(date +%Y%m%d_%H%M%S).log

# python3 -u -m model.main model/configs/full_run_optimised_agl_novpp.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_optimised_agl_novpp_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_self_consumption_agl_novpp.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_self_consumption_agl_novpp_$(date +%Y%m%d_%H%M%S).log

# python3 -u -m model.main model/configs/full_run_optimised_origin_max.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_optimised_origin_max_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_optimised_origin_lite.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_optimised_origin_lite_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_optimised_ea.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_optimised_ea_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_optimised_agl.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_optimised_agl_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_optimised_globird.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_optimised_globird_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_optimised_amber.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_optimised_amber_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_self_consumption_origin_max.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_self_consumption_origin_max_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_self_consumption_origin_lite.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_self_consumption_origin_lite_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_self_consumption_ea.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_self_consumption_ea_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_self_consumption_agl.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_self_consumption_agl_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_self_consumption_globird.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_self_consumption_globird_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_self_consumption_amber.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_self_consumption_amber_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_solar_only_origin_max.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_solar_only_origin_max_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_solar_only_origin_starter.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_solar_only_origin_starter_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_solar_only_ea.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_solar_only_ea_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_solar_only_agl.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_solar_only_agl_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_solar_only_globird.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_solar_only_globird_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_solar_only_amber.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_solar_only_amber_$(date +%Y%m%d_%H%M%S).log
# python3 -u -m model.main model/configs/full_run_no_cer_amber.yaml --validate-only --save-flows --monthly-bills --workers 6 2>&1 | tee logs/full_run_no_cer_amber_$(date +%Y%m%d_%H%M%S).log
