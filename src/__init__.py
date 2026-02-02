"""
src package
===========
Production code for Blue Book for Bulldozers ML pipeline.
"""

__version__ = "0.1.0"
__author__ = "Matteo Postiferi"

# Optional: Make imports easier
from .config import *
from .preprocessing import load_data, prepare_data
# from .model import train_models  # When ready
# from .evaluation import evaluate_model  # When ready