#!/usr/bin/env python3
"""Thin wrapper: runs experiments/run_experiment.py (unified script)."""

import os
import sys

_EXPERIMENTS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, _EXPERIMENTS)
os.chdir(_EXPERIMENTS)

import run_experiment as main_module

if __name__ == "__main__":
    main_module.main()
