clear
rm -rf fl_logs/*
PYTHONUNBUFFERED=1 flwr run . 2>&1 | tee fl_logs/simulation_full_output.log
