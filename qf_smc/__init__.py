"""qf_smc — QUANTFLOW Smart Money Concepts Long Scanner"""

__version__ = "1.0.0"

# Re-export the main public API for easy imports from app.py
from qf_smc.scanner import (
    MODE_CONFIG,
    run_scan,
    scan_one_symbol,
)
