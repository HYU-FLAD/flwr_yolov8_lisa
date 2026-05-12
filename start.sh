clear
rm -rf fl_logs_nc2/*
PYTHONUNBUFFERED=1 flwr run . local-simulation --stream 2>&1 | tee nc2_clean_20clients.log
rm -rf fl_logs_nc2/client_node*