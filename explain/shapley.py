# shapley.py
# DyGLib-Explainability
# Justin Hoang
# Updated for TGB-based temporal GNN models

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
import warnings
import math
from itertools import combinations
from tqdm import tqdm
from typing import Dict, List, Optional


class ShapleyExplainer:
    """
    SHAP Explainer.
    """

    def __init__(
        self, 
    ):
        pass