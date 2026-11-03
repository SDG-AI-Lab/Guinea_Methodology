"""riskmapjnr's local deforestation rate on a forest-change raster (runs in .venv_tools).
Usage: fcc_file out_dir years ws [ws ...]

Writes the package's ldefrate_ws<ws>.tif for each window size, for the code check of
tool_comparison.check_against_package.
"""
import os
import sys

import riskmapjnr as rmj

fcc_file, out_dir, years = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(out_dir, exist_ok=True)
for ws in map(int, sys.argv[4:]):
    rmj.local_defor_rate(fcc_file=fcc_file, defor_values=1, ldefrate_file=os.path.join(out_dir, f"ldefrate_ws{ws}.tif"),
                         win_size=ws, time_interval=years, verbose=False)
    print("done", ws, flush=True)
