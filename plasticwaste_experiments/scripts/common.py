"""Paths shared by the plastic-waste experiments. The deployed tool is imported read-only."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(r"C:\Users\ASUS\Guinea_Project\Land Use-Deforestation")
TOOL = PROJECT / "plasticwaste_tool"
MARIDA = Path(r"D:\guinea_data\marida")
RASTERS = Path(r"D:\gee_cache\cache\rasters")
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
