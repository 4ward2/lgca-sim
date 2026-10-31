"""A5 code: multiplication tasks (M1D, M2D), K=16 hand oracle, K=2 search. New code only; imports sim/ read-only."""
import os
import sys

SIM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SIM_DIR not in sys.path:
    sys.path.insert(0, SIM_DIR)
