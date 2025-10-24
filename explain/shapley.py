# shapley.py
# DyGLib-Explainability
# Justin Hoang
# 10/24/2025

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import warnings
warnings.filterwarnings('ignore')

from typing import List, Tuple, Dict, Any
from itertools import combinations
from tqdm import tqdm

from utils.DataLoader import get_link_prediction_data, Data
from utils.utils import get_neighbor_sampler, convert_to_gpu
from utils.load_configs import get_link_prediction_args
from utils.EarlyStopping import EarlyStopping
from models.DyGFormer import DyGFormer
from models.modules import MergeLayer

class ShapleyExplainer:
    def __init__(self, model: nn.Module, device: str = "cpu", num_samples: int = 100):
        self.model = model
        self.device = device
        self.num_samples = num_samples
        self.model.eval()
        
    def compute_shapley_values(
        self, 
        src_node_ids: np.ndarray, 
        dst_node_ids: np.ndarray, 
        node_interact_times: np.ndarray, 
        neighbor_sampler, 
        temporal_importance: bool = True
    ):
        pass
    
    def _get_baseline_prediction(
        self,
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        neighbor_sampler
    ):
        pass
    
    def _get_zero_embeddings(
        self,
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        neighbor_sampler
    ):
        pass
    
    def _compute_temporal_shapley(
        self,
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        neighbor_sampler,
        baseline_pred
    ):
        """
        Compute temporal Shapley values.

        Args:
            src_node_ids (np.ndarray): _description_
            dst_node_ids (np.ndarray): _description_
            node_interact_times (np.ndarray): _description_
            neighbor_sampler (_type_): _description_
            baseline_pred (_type_): _description_
        """
        
    
    def _get_prediction_at_time(
        self, 
        src_id: int, 
        dst_id: int, 
        time: float
    ):
        """
        Get prediction at specific time point.

        Args:
            src_id: int
            dst_id: int
            time: float

        Returns:
            pred: Tensor
        """
        src_emb, dst_emb = self.model[0].compute_src_dst_node_temporal_embeddings(
            np.array([src_id]), np.array([dst_id]), np.array([time])
        )
        pred = self.model[1](src_emb, dst_emb)
        return pred