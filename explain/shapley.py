# shapley.py
# DyGLib-Explainability
# Justin Hoang
# 10/24/2025

import torch
import torch.nn as nn
import numpy as np

import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
from typing import Dict


class ShapleyExplainer:
    def __init__(self, model: nn.Module, device: str = "cpu", num_samples: int = 100):
        self.model: nn.Module = model
        self.device: str = device
        self.num_samples: int = num_samples
        self.model.eval()

    def compute_shapley_values(
        self,
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        temporal_importance: bool = True,
    ) -> Dict:
        """
        Compute Shapley values.

        Args:
            src_node_ids: np.ndarray
            dst_node_ids: np.ndarray
            node_interact_times: np.ndarray
            temporal_importance: bool

        Returns:
            results: Dict
        """
        results = {}

        with torch.no_grad():

            if temporal_importance:
                print("Computing temporal Shapley values...")
                temporal_shapley = self._compute_temporal_shapley(
                    src_node_ids, dst_node_ids, node_interact_times
                )
                results["temporal_shapley"] = temporal_shapley

        return results

    def _compute_temporal_shapley(
        self,
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        node_interact_times: np.ndarray,
    ) -> np.ndarray:
        """
        Compute temporal Shapley values.

        Args:
            src_node_ids: np.ndarray
            dst_node_ids: np.ndarray
            node_interact_times: np.ndarray

        Returns:
            temporal_shapley: np.ndarray
        """
        batch_size = len(src_node_ids)
        temporal_shapley = np.zeros(batch_size)

        for i in tqdm(
            range(batch_size), desc="Computing temporal features SHAP values"
        ):
            current_time = node_interact_times[i]

            # prediction with temporal information
            pred_with_time = self._get_prediction_at_time(
                src_node_ids[i], dst_node_ids[i], current_time
            )
            
            # prediction without temporal information
            pred_without_time = self._get_prediction_at_time(
                src_node_ids[i], dst_node_ids[i], min(node_interact_times)
            )
            
            temporal_shapley[i] = (pred_with_time - pred_without_time).item()

        return temporal_shapley

    def _get_prediction_at_time(
        self, src_id: int, dst_id: int, time: float
    ) -> torch.Tensor:
        """
        Get prediction at specific time point.

        Args:
            src_id: int
            dst_id: int
            time: float

        Returns:
            pred: Tensor
        """
        # get temporal embeddings at the specific time point
        src_emb, dst_emb = self.model[0].compute_src_dst_node_temporal_embeddings(
            np.array([src_id]), np.array([dst_id]), np.array([time])
        )
        pred = self.model[1](src_emb, dst_emb)
        return pred
