#!/bin/sh
# Runs the whole waste-site benchmark (scripts 06-09) in order. 05 is already done (pool + crops).
# 06 skips features that exist. A first table (09) is written before the slowest run
# (Guatemala protocol at 300 m, 1024 px inputs), and again at the end.
# Uses the GPU environment of the national scan. Logs: outputs/benchmark/logs/
PY="/e/SDG AI Lab/Guinea/national_scan_2026/venv_gpu/Scripts/python.exe"
cd "$(dirname "$0")"; L=outputs/benchmark/logs; mkdir -p $L
export PYTHONWARNINGS=ignore HF_HUB_DISABLE_SYMLINKS_WARNING=1 HF_HUB_DISABLE_PROGRESS_BARS=1
report() { "$PY" 09_benchmark_report.py main > $L/09_main.log 2>&1; "$PY" 09_benchmark_report.py sensitivity > $L/09_sensitivity.log 2>&1; }
"$PY" 06_frozen_features.py > $L/06.log 2>&1 || exit 1
"$PY" 07_linear_probe_benchmark.py main > $L/07_main.log 2>&1
"$PY" 07_linear_probe_benchmark.py sensitivity > $L/07_sensitivity.log 2>&1
for w in 80 150 300; do "$PY" 08_trained_methods.py finetune $w > $L/08_finetune_$w.log 2>&1; done
for w in 80 150; do "$PY" 08_trained_methods.py aw_protocol $w > $L/08_aw_protocol_$w.log 2>&1; done
report
"$PY" 08_trained_methods.py aw_protocol 300 > $L/08_aw_protocol_300.log 2>&1
report
echo done > $L/ALL_DONE
