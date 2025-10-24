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
from typing import Dict, Tuple
from utils.utils import NeighborSampler


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
        neighbor_sampler: NeighborSampler,
        temporal_importance: bool = True,
    ) -> Dict:
        """
        Compute Shapley values.

        Args:
            src_node_ids: np.ndarray
            dst_node_ids: np.ndarray
            node_interact_times: np.ndarray
            neighbor_sampler: NeighborSampler
            temporal_importance: bool

        Returns:
            results: Dict
        """
        results = {}
        baseline_pred = self._get_baseline_prediction(src_node_ids)

        with torch.no_grad():

            if temporal_importance:
                print("Computing temporal Shapley values...")
                temporal_shapley = self._compute_temporal_shapley(
                    src_node_ids, dst_node_ids, node_interact_times, baseline_pred
                )
                results["temporal_shapley"] = temporal_shapley

        return results

    def _get_baseline_prediction(self, src_node_ids: np.ndarray) -> torch.Tensor:
        """
        Get baseline prediction.

        Args:
            src_node_ids: np.ndarray

        Returns:
            baseline_pred: Tensor
        """
        zero_src_emb, zero_dst_emb = self._get_zero_embeddings(src_node_ids)
        baseline_pred = self.model[1](zero_src_emb, zero_dst_emb)
        return baseline_pred

    def _get_zero_embeddings(
        self, src_node_ids: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get zero embeddings.

        Args:
            src_node_ids: np.ndarray

        Returns:
            zero_emb: Tensor
        """
        batch_size = len(src_node_ids)
        zero_emb = torch.zeros(
            batch_size, self.model[0].node_feat_dim, device=self.device
        )
        return zero_emb, zero_emb

    def _compute_temporal_shapley(
        self,
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        baseline_pred: torch.Tensor,
    ) -> np.ndarray:
        """
        Compute temporal Shapley values.

        Args:
            src_node_ids: np.ndarray
            dst_node_ids: np.ndarray
            node_interact_times: np.ndarray
            baseline_pred: torch.Tensor

        Returns:
            temporal_shapley: np.ndarray
        """
        batch_size = len(src_node_ids)
        temporal_shapley = np.zeros(batch_size)

        for i in tqdm(
            range(batch_size), desc="Computing temporal features SHAP values"
        ):
            current_time = node_interact_times[i]

            time_variations = np.linspace(current_time * 0.5, current_time * 1.5, 10)

            shapley_sum = 0.0
            for time_var in time_variations:
                pred_time = self._get_prediction_at_time(
                    src_node_ids[i], dst_node_ids[i], time_var
                )
                pred_current = self._get_prediction_at_time(
                    src_node_ids[i], dst_node_ids[i], current_time
                )

                shapley_sum += (pred_current - baseline_pred).item()

            temporal_shapley[i] = shapley_sum / len(time_variations)

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
        src_emb, dst_emb = self.model[0].compute_src_dst_node_temporal_embeddings(
            np.array([src_id]), np.array([dst_id]), np.array([time])
        )
        pred = self.model[1](src_emb, dst_emb)
        return pred
