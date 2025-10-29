# shapley.py
# DyGLib-Explainability
# Justin Hoang
# 10/24/2025

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
import warnings

from tqdm import tqdm
from typing import Dict


class ShapleyExplainer:
    """
    Shapley explainer for the DyGFormer temporal graph neural network prediction model.

    Args:
        model: nn.Module
        device: str
        num_samples: int
    """

    def __init__(self, model: nn.Module, device: str = "cpu", num_samples: int = 100):
        self.model: nn.Module = model
        self.device: str = device
        self.num_samples: int = num_samples

        warnings.filterwarnings("ignore")
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger()

        self.model.eval()

    def get_summary_statistics(self, results: Dict) -> pd.DataFrame:
        """
        Get summary statistics of the Shapley values.

        Args:
            results (Dict): Dictionary containing the results of the Shapley values computation.

        Returns:
            pd.DataFrame: DataFrame containing the summary statistics of the Shapley values.
        """
        temporal_shapley = results["temporal_shapley"]

        if temporal_shapley is None:
            return None

        summary = pd.DataFrame(
            {
                "Feature": ["Temporal Information"],
                "Mean_Shapley": [np.mean(temporal_shapley)],
                "Std_Shapley": [np.std(temporal_shapley)],
                "Min_Shapley": [np.min(temporal_shapley)],
                "Max_Shapley": [np.max(temporal_shapley)],
                "Median_Shapley": [np.median(temporal_shapley)],
                "Mean_Absolute_Shapley": [np.mean(np.abs(temporal_shapley))],
            }
        )

        return summary

    def visualize_top_interactions(
        self,
        results: Dict,
        top_k: int = 10,
    ) -> pd.DataFrame:
        """
        Visualize top interactions by temporal Shapley values.

        Args:
            results (Dict): Dictionary containing the results of the Shapley values computation.
            top_k (int, optional): Number of top interactions to visualize. Defaults to 10.

        Returns:
            pd.DataFrame: DataFrame containing the top interactions by temporal Shapley values.
        """
        df = results.get("temporal_shapley_df")

        if df is None:
            return None

        return df.head(top_k)

    def compute_shapley_values(
        self,
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        temporal_importance: bool = True,
        return_df: bool = True,
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

                if return_df:
                    results["temporal_shapley_df"] = self._format_temporal_results(
                        src_node_ids,
                        dst_node_ids,
                        node_interact_times,
                        temporal_shapley,
                    )

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

        baseline_time = np.median(node_interact_times)

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
                src_node_ids[i], dst_node_ids[i], baseline_time
            )

            temporal_shapley[i] = (pred_with_time - pred_without_time).item()

        return temporal_shapley

    def _get_prediction_at_time(
        self, src_id: int, dst_id: int, time: float
    ) -> torch.Tensor:
        """
        Get graph prediction at specific time point.

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
        pred = self.model[1](src_emb, dst_emb).squeeze().sigmoid()
        return pred

    def _format_temporal_results(
        self,
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        temporal_shapley: np.ndarray,
    ) -> pd.DataFrame:
        """
        Format temporal results.

        Args:
            src_node_ids (np.ndarray): Source node IDs.
            dst_node_ids (np.ndarray): Destination node IDs.
            node_interact_times (np.ndarray): Node interaction times.
            temporal_shapley (np.ndarray): Temporal Shapley values.

        Returns:
            pd.DataFrame: DataFrame containing the temporal results.
        """
        df = pd.DataFrame(
            {
                "Source_Node": src_node_ids,
                "Destination_Node": dst_node_ids,
                "Interaction-Time": node_interact_times,
                "Temporal_Shapley_Value": temporal_shapley,
                "Temporal_Importance": np.abs(temporal_shapley),
                "Temporal_Effect": [
                    "Positive" if x > 0 else "Negative" for x in temporal_shapley
                ],
            }
        )

        df = df.sort_values("Temporal_Importance", ascending=False).reset_index(
            drop=True
        )

        return df
