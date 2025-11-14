# shapley.py
# DyGLib-Explainability
# Justin Hoang
# 10/28/2025

import shap
import torch
import torch.nn as nn
import logging
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Any


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()


class ShapleyExplainer:
    """
    SHAP Explainer for Temporal Graph Neural Networks.
    Uses KernelSHAP to explain link predictions in temporal graphs.
    """

    def __init__(
        self,
        model: nn.Module,
        node_raw_features: np.ndarray,
        edge_raw_features: np.ndarray,
        num_samples: int = 100,
        sample_ratio: float = 0.1,
        nsamples: int = 100,
        device: str = "cpu",
    ):
        """
        Initialize the Shapley Explainer.

        Args:
            model: The trained temporal GNN model (Sequential with backbone + predictor)
            node_raw_features: Raw node features
            edge_raw_features: Raw edge features
            num_samples: Number of samples for background data in KernelSHAP (default: 100)
                        Lower = faster but less accurate baseline
            sample_ratio: Ratio of test data to explain (default: 0.1 = 10%)
                         Lower = faster but less coverage
            nsamples: Number of times to re-evaluate model per explanation (default: 100)
                     Lower = much faster but less accurate SHAP values
                     Try 50 for 2x speedup, 25 for 4x speedup
            device: Device to run computations on
        """
        self.model = model
        self.node_raw_features = node_raw_features
        self.edge_raw_features = edge_raw_features
        self.num_samples = num_samples
        self.sample_ratio = sample_ratio
        self.nsamples = nsamples
        self.device = device

        self.model.eval()

        self.shap_values = None
        self.shap_explainer = None
        self.feature_names = None
        self.sampled_indices = None

        logger.info(f"Initialized ShapleyExplainer:")
        logger.info(f"  - num_samples (background): {num_samples}")
        logger.info(f"  - sample_ratio (explain): {sample_ratio}")
        logger.info(f"  - nsamples (model evals): {nsamples}")
        logger.info(
            f"  - Estimated model calls: ~{int(nsamples * sample_ratio * 10000)}"
        )

    def _create_feature_matrix(
        self,
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        edge_ids: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Create a feature matrix from temporal graph interactions.
        Uses aggregated statistics instead of individual features for better interpretability
        and to avoid huge SHAP values from feature scale mismatches.
        
        Args:
            src_node_ids: Source node IDs
            dst_node_ids: Destination node IDs
            node_interact_times: Interaction timestamps
            edge_ids: Edge IDs (optional)

        Returns:
            feature_matrix: Feature matrix
            feature_names: Feature names
        """
        num_interactions = len(src_node_ids)

        feature_list = []
        feature_names = []

        # Source node features - aggregate statistics
        src_features = self.node_raw_features[src_node_ids]
        if len(src_features.shape) == 1:
            src_features = src_features.reshape(-1, 1)

        # Compute aggregated statistics for source node
        src_mean = src_features.mean(axis=1, keepdims=True)
        src_std = src_features.std(axis=1, keepdims=True)
        src_max = src_features.max(axis=1, keepdims=True)
        src_min = src_features.min(axis=1, keepdims=True)

        # Normalize each statistic to [0, 1] range
        src_mean_norm = (src_mean - src_mean.min()) / (
            src_mean.max() - src_mean.min() + 1e-10
        )
        src_std_norm = (src_std - src_std.min()) / (
            src_std.max() - src_std.min() + 1e-10
        )
        src_max_norm = (src_max - src_max.min()) / (
            src_max.max() - src_max.min() + 1e-10
        )
        src_min_norm = (src_min - src_min.min()) / (
            src_min.max() - src_min.min() + 1e-10
        )

        feature_list.extend([src_mean_norm, src_std_norm, src_max_norm, src_min_norm])
        feature_names.extend(
            ["src_node_mean", "src_node_std", "src_node_max", "src_node_min"]
        )

        # Destination node features - same aggregation
        dst_features = self.node_raw_features[dst_node_ids]
        if len(dst_features.shape) == 1:
            dst_features = dst_features.reshape(-1, 1)

        dst_mean = dst_features.mean(axis=1, keepdims=True)
        dst_std = dst_features.std(axis=1, keepdims=True)
        dst_max = dst_features.max(axis=1, keepdims=True)
        dst_min = dst_features.min(axis=1, keepdims=True)

        dst_mean_norm = (dst_mean - dst_mean.min()) / (
            dst_mean.max() - dst_mean.min() + 1e-10
        )
        dst_std_norm = (dst_std - dst_std.min()) / (
            dst_std.max() - dst_std.min() + 1e-10
        )
        dst_max_norm = (dst_max - dst_max.min()) / (
            dst_max.max() - dst_max.min() + 1e-10
        )
        dst_min_norm = (dst_min - dst_min.min()) / (
            dst_min.max() - dst_min.min() + 1e-10
        )

        feature_list.extend([dst_mean_norm, dst_std_norm, dst_max_norm, dst_min_norm])
        feature_names.extend(
            ["dst_node_mean", "dst_node_std", "dst_node_max", "dst_node_min"]
        )

        # Edge features - aggregate if available
        if edge_ids is not None and len(self.edge_raw_features) > 0:
            edge_features = self.edge_raw_features[edge_ids]
            if len(edge_features.shape) == 1:
                edge_features = edge_features.reshape(-1, 1)

            edge_mean = edge_features.mean(axis=1, keepdims=True)
            edge_std = edge_features.std(axis=1, keepdims=True)
            edge_max = edge_features.max(axis=1, keepdims=True)

            edge_mean_norm = (edge_mean - edge_mean.min()) / (
                edge_mean.max() - edge_mean.min() + 1e-10
            )
            edge_std_norm = (edge_std - edge_std.min()) / (
                edge_std.max() - edge_std.min() + 1e-10
            )
            edge_max_norm = (edge_max - edge_max.min()) / (
                edge_max.max() - edge_max.min() + 1e-10
            )

            feature_list.extend([edge_mean_norm, edge_std_norm, edge_max_norm])
            feature_names.extend(["edge_mean", "edge_std", "edge_max"])

        # Temporal features - normalized timestamp
        time_min = node_interact_times.min()
        time_max = node_interact_times.max()
        time_features = (node_interact_times - time_min) / (time_max - time_min + 1e-10)
        time_features = time_features.reshape(-1, 1)
        feature_list.append(time_features)
        feature_names.append("timestamp")

        # Node ID features (normalized) - can indicate structural importance
        src_id_normalized = (src_node_ids / src_node_ids.max()).reshape(-1, 1)
        dst_id_normalized = (dst_node_ids / dst_node_ids.max()).reshape(-1, 1)
        feature_list.extend([src_id_normalized, dst_id_normalized])
        feature_names.extend(["src_node_id", "dst_node_id"])

        # Concatenate all features
        feature_matrix = np.concatenate(feature_list, axis=1)

        logger.info(
            f"Created feature matrix with {feature_matrix.shape[1]} aggregated features (reduced from 519 individual features)"
        )

        return feature_matrix, feature_names

    def _create_model_wrapper(
        self,
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        edge_ids: Optional[np.ndarray] = None,
    ):
        """
        Create a wrapper function for the model that SHAP can use.
        The wrapper takes the feature matrix and returns predictions.

        Args:
            src_node_ids: Source node IDs
            dst_node_ids: Destination node IDs
            node_interact_times: Interaction timestamps
            edge_ids: Edge IDs (optional)

        Returns:
            wrapper_fn: Function that takes feature matrix and returns predictions
        """

        def wrapper_fn(feature_matrix: np.ndarray) -> np.ndarray:
            """
            Wrapper function that converts feature matrix back to graph data
            and runs the model. Processes in batches for efficiency.
            """
            batch_size = feature_matrix.shape[0]

            # Handle SHAP's synthetic samples by mapping to actual data indices
            indices = np.array(
                [i if i < len(src_node_ids) else 0 for i in range(batch_size)]
            )

            # Get batched inputs
            src_ids_batch = src_node_ids[indices]
            dst_ids_batch = dst_node_ids[indices]
            times_batch = node_interact_times[indices]

            with torch.no_grad():
                # Get the dynamic backbone (first module in Sequential)
                dynamic_backbone = self.model[0]

                # Compute embeddings using the backbone (batched)
                src_emb, dst_emb = (
                    dynamic_backbone.compute_src_dst_node_temporal_embeddings(
                        src_node_ids=src_ids_batch,
                        dst_node_ids=dst_ids_batch,
                        node_interact_times=times_batch,
                    )
                )

                # Get the link predictor (second module in Sequential)
                link_predictor = self.model[1]

                # Make predictions (batched)
                predictions = link_predictor(src_emb, dst_emb).squeeze()

                # Apply sigmoid to convert logits to probabilities [0, 1]
                predictions = torch.sigmoid(predictions)

                # Convert to numpy
                if predictions.dim() == 0:  # Single value
                    predictions = predictions.unsqueeze(0)
                predictions = predictions.cpu().numpy()

            return predictions

        return wrapper_fn

    def compute_shapley_values(
        self,
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        edge_ids: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compute SHAP values for the temporal graph model using KernelSHAP.

        Args:
            src_node_ids: Source node IDs
            dst_node_ids: Destination node IDs
            node_interact_times: Interaction timestamps
            edge_ids: Edge IDs (optional)
            labels: Ground truth labels (optional)
            save_path: Path to save results (optional)

        Returns:
            results: Dictionary containing SHAP values and metadata
        """
        logger.info("Computing SHAP values using KernelSHAP...")

        # Create feature matrix
        logger.info("Creating feature matrix from temporal graph data...")
        feature_matrix, feature_names = self._create_feature_matrix(
            src_node_ids, dst_node_ids, node_interact_times, edge_ids
        )
        self.feature_names = feature_names

        # Sample data for efficiency
        num_total = feature_matrix.shape[0]
        num_explain = max(int(num_total * self.sample_ratio), 10)  # At least 10 samples

        # Sample indices for explanation (from the end of data, which is test set)
        np.random.seed(42)
        self.sampled_indices = np.random.choice(
            range(num_total - num_explain, num_total),
            size=min(num_explain, num_total),
            replace=False,
        )

        logger.info(
            f"Sampled {len(self.sampled_indices)} interactions out of {num_total} for explanation"
        )

        # Create background data (from earlier interactions)
        background_indices = np.random.choice(
            range(0, num_total - num_explain),
            size=min(self.num_samples, num_total - num_explain),
            replace=False,
        )
        background_data = feature_matrix[background_indices]

        # Data to explain
        explain_data = feature_matrix[self.sampled_indices]

        # Create model wrapper
        logger.info("Creating model wrapper for SHAP...")
        model_wrapper = self._create_model_wrapper(
            src_node_ids, dst_node_ids, node_interact_times, edge_ids
        )

        # Initialize KernelSHAP explainer
        logger.info("Initializing KernelSHAP explainer...")
        self.shap_explainer = shap.KernelExplainer(
            model=model_wrapper,
            data=background_data,
            link="identity",
        )

        # Compute SHAP values
        logger.info("Computing SHAP values (this may take a while)...")
        logger.info(f"  - Explaining {len(self.sampled_indices)} samples")
        logger.info(f"  - Using {self.nsamples} model evaluations per sample")
        logger.info(
            f"  - Total model calls: ~{len(self.sampled_indices) * self.nsamples}"
        )

        # Choose regularization based on sample/feature ratio
        num_features = explain_data.shape[1]
        if self.num_samples < num_features:
            l1_reg = "num_features(10)"
            logger.info(
                f"  - Using l1_reg='num_features(10)' (samples={self.num_samples} < features={num_features})"
            )
        else:
            l1_reg = "aic"  # Use AIC when we have enough samples
            logger.info(f"  - Using l1_reg='aic'")

        self.shap_values = self.shap_explainer.shap_values(
            explain_data,
            nsamples=self.nsamples,  # Number of times to re-evaluate the model
            l1_reg=l1_reg,
        )

        logger.info(f"SHAP values computed! Shape: {self.shap_values.shape}")

        # Prepare results
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        feature_importance = {
            fname: float(importance)
            for fname, importance in zip(feature_names, mean_abs_shap)
        }

        # Sort features by importance and filter out zero-importance features
        sorted_features = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )

        # Filter out zero-importance features (with small threshold to handle floating point precision)
        non_zero_features = [(f, imp) for f, imp in sorted_features if imp > 1e-10]
        zero_features = [f for f, imp in sorted_features if imp <= 1e-10]

        # Filter feature_importance dictionary
        filtered_feature_importance = {
            fname: imp for fname, imp in feature_importance.items() if imp > 1e-10
        }

        # Create comprehensive summary (moved from console to JSON)
        results = {
            "summary": {
                "base_value": float(self.shap_explainer.expected_value),
                "num_features": len(feature_names),
                "num_samples_explained": len(self.sampled_indices),
                "background_size": len(background_indices),
                "top_10_features": dict(non_zero_features[:10]),
                "top_20_features": dict(non_zero_features[:20]),
                "zero_importance_features": zero_features,  # List of features with zero importance
                "min_shap_value": float(self.shap_values.min()),
                "max_shap_value": float(self.shap_values.max()),
                "mean_abs_shap_value": float(mean_abs_shap.mean()),
            },
            "feature_names": feature_names,
            "feature_importance": filtered_feature_importance,  # Only non-zero features
            "sampled_indices": self.sampled_indices.tolist(),
            "shap_values_shape": list(self.shap_values.shape),
            # Note: Not including full shap_values array to keep JSON manageable
            # Use visualization plots to see detailed SHAP values
        }

        # Log basic completion message (detailed summary now in JSON)
        logger.info(f"✓ SHAP analysis complete")
        logger.info(f"  Base value: {results['summary']['base_value']:.4f}")
        if non_zero_features:
            logger.info(
                f"  Top feature: {non_zero_features[0][0]} (importance: {non_zero_features[0][1]:.4f})"
            )
            logger.info(
                f"  Non-zero features: {len(non_zero_features)}/{len(feature_names)}"
            )
        else:
            logger.warning("  No features with non-zero importance found!")

        # Save results if path provided
        if save_path:
            self._save_results(results, save_path)

        return results

    def _save_results(self, results: Dict[str, Any], save_path: str):
        """
        Save SHAP results to JSON file.

        Args:
            results: Results dictionary
            save_path: Path to save the JSON file
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {save_path}")

    def create_visualizations(
        self,
        save_dir: str,
        dataset_name: str,
        model_name: str,
    ):
        """
        Create comprehensive SHAP visualizations.

        Args:
            save_dir: Directory to save plots
            dataset_name: Name of the dataset
            model_name: Name of the model
        """
        if self.shap_values is None:
            raise ValueError("Must compute SHAP values before creating visualizations!")

        logger.info(f"Creating SHAP visualizations in {save_dir}...")
        os.makedirs(save_dir, exist_ok=True)

        # Filter out zero-importance features for visualizations
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        non_zero_mask = mean_abs_shap > 1e-10
        filtered_shap_values = self.shap_values[:, non_zero_mask]
        filtered_feature_names = [
            name for name, keep in zip(self.feature_names, non_zero_mask) if keep
        ]

        num_filtered = len(self.feature_names) - len(filtered_feature_names)
        if num_filtered > 0:
            logger.info(
                f"Filtering out {num_filtered} zero-importance features from visualizations"
            )

        # Prepare data for SHAP plots
        # We need to reconstruct the feature matrix for the explained samples

        # 1. Summary plot (beeswarm) - Overall feature importance
        logger.info("Creating beeswarm plot...")
        try:
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                filtered_shap_values,
                features=None,  # We don't have the actual feature values here
                feature_names=filtered_feature_names,
                plot_type="dot",
                show=False,
                max_display=20,
            )
            plt.title(
                f"SHAP Summary Plot - {model_name} on {dataset_name}",
                fontsize=14,
                pad=20,
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(save_dir, "shap_beeswarm.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
            logger.info("Beeswarm plot saved")
        except Exception as e:
            logger.warning(f"Could not create beeswarm plot: {e}")

        # 2. Bar plot - Mean absolute SHAP values
        logger.info("Creating bar plot...")
        try:
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                filtered_shap_values,
                feature_names=filtered_feature_names,
                plot_type="bar",
                show=False,
                max_display=20,
            )
            plt.title(
                f"Feature Importance - {model_name} on {dataset_name}",
                fontsize=14,
                pad=20,
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(save_dir, "shap_bar.png"), dpi=300, bbox_inches="tight"
            )
            plt.close()
            logger.info("Bar plot saved")
        except Exception as e:
            logger.warning(f"Could not create bar plot: {e}")

        # 3. Waterfall plots for individual predictions (first 5)
        logger.info("Creating waterfall plots for individual predictions...")
        num_waterfall = min(5, self.shap_values.shape[0])

        for i in range(num_waterfall):
            try:
                plt.figure(figsize=(12, 8))

                # Create Explanation object for waterfall plot (filtered)
                filtered_values = self.shap_values[i][non_zero_mask]
                explanation = shap.Explanation(
                    values=filtered_values,
                    base_values=self.shap_explainer.expected_value,
                    feature_names=filtered_feature_names,
                )

                shap.plots.waterfall(explanation, show=False, max_display=15)
                plt.title(f"SHAP Waterfall Plot - Sample {i+1}", fontsize=14, pad=20)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(save_dir, f"shap_waterfall_sample_{i+1}.png"),
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()
            except Exception as e:
                logger.warning(f"Could not create waterfall plot {i+1}: {e}")

        logger.info(f"Created {num_waterfall} waterfall plots")

        # 4. Heatmap of SHAP values
        logger.info("Creating heatmap...")
        try:
            plt.figure(figsize=(14, 10))

            # Show heatmap for top features and samples (using filtered data)
            num_samples_show = min(50, filtered_shap_values.shape[0])
            num_features_show = min(20, filtered_shap_values.shape[1])

            # Get top features by mean absolute SHAP value (from filtered data)
            mean_abs_shap_filtered = np.abs(filtered_shap_values).mean(axis=0)
            top_feature_indices = np.argsort(mean_abs_shap_filtered)[
                -num_features_show:
            ][::-1]

            shap_subset = filtered_shap_values[:num_samples_show, top_feature_indices]
            feature_names_subset = [
                filtered_feature_names[i] for i in top_feature_indices
            ]

            plt.imshow(
                shap_subset.T,
                aspect="auto",
                cmap="RdBu_r",
                vmin=-np.abs(shap_subset).max(),
                vmax=np.abs(shap_subset).max(),
            )
            plt.colorbar(label="SHAP value")
            plt.xlabel("Sample Index", fontsize=12)
            plt.ylabel("Feature", fontsize=12)
            plt.yticks(
                range(len(feature_names_subset)), feature_names_subset, fontsize=8
            )
            plt.title(
                f"SHAP Values Heatmap - {model_name} on {dataset_name}",
                fontsize=14,
                pad=20,
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(save_dir, "shap_heatmap.png"), dpi=300, bbox_inches="tight"
            )
            plt.close()
            logger.info("Heatmap saved")
        except Exception as e:
            logger.warning(f"Could not create heatmap: {e}")

        # 5. Feature importance comparison plot
        logger.info("Creating feature importance comparison plot...")
        try:
            mean_abs_shap_filtered = np.abs(filtered_shap_values).mean(axis=0)
            top_n = min(15, len(filtered_feature_names))
            top_indices = np.argsort(mean_abs_shap_filtered)[-top_n:][::-1]

            plt.figure(figsize=(12, 8))
            plt.barh(
                range(len(top_indices)),
                mean_abs_shap_filtered[top_indices],
                color="steelblue",
                alpha=0.8,
            )
            plt.yticks(
                range(len(top_indices)),
                [filtered_feature_names[i] for i in top_indices],
                fontsize=10,
            )
            plt.xlabel("Mean |SHAP value|", fontsize=12)
            plt.title(
                f"Top {top_n} Features by SHAP Importance - {model_name} on {dataset_name}",
                fontsize=14,
                pad=20,
            )
            plt.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                os.path.join(save_dir, "shap_feature_importance.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
            logger.info("Feature importance plot saved")
        except Exception as e:
            logger.warning(f"Could not create feature importance plot: {e}")

        # 6. Overall summary statistics plot
        logger.info("Creating overall statistics plot...")
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))

            # Distribution of SHAP values (filtered)
            axes[0, 0].hist(
                filtered_shap_values.flatten(), bins=50, color="steelblue", alpha=0.7
            )
            axes[0, 0].set_xlabel("SHAP Value", fontsize=11)
            axes[0, 0].set_ylabel("Frequency", fontsize=11)
            axes[0, 0].set_title(
                "Distribution of SHAP Values (Non-Zero Features)", fontsize=12
            )
            axes[0, 0].grid(alpha=0.3)

            # Mean absolute SHAP value per feature (filtered)
            mean_abs_shap_filtered = np.abs(filtered_shap_values).mean(axis=0)
            top_10_count = min(10, len(filtered_feature_names))
            top_10_indices = np.argsort(mean_abs_shap_filtered)[-top_10_count:][::-1]
            axes[0, 1].barh(
                range(len(top_10_indices)),
                mean_abs_shap_filtered[top_10_indices],
                color="coral",
                alpha=0.7,
            )
            axes[0, 1].set_yticks(range(len(top_10_indices)))
            axes[0, 1].set_yticklabels(
                [filtered_feature_names[i] for i in top_10_indices], fontsize=9
            )
            axes[0, 1].set_xlabel("Mean |SHAP|", fontsize=11)
            axes[0, 1].set_title("Top 10 Features by Mean |SHAP|", fontsize=12)
            axes[0, 1].grid(axis="x", alpha=0.3)

            # Positive vs negative SHAP values (filtered)
            mean_shap_filtered = filtered_shap_values.mean(axis=0)
            axes[1, 0].barh(
                range(len(mean_shap_filtered)),
                mean_shap_filtered,
                color=["green" if x > 0 else "red" for x in mean_shap_filtered],
                alpha=0.6,
            )
            axes[1, 0].axvline(x=0, color="black", linestyle="--", linewidth=1)
            axes[1, 0].set_xlabel("Mean SHAP Value", fontsize=11)
            axes[1, 0].set_ylabel("Feature Index", fontsize=11)
            axes[1, 0].set_title(
                "Mean SHAP Values (Green=Positive, Red=Negative)", fontsize=12
            )
            axes[1, 0].grid(alpha=0.3)

            # Variance of SHAP values per feature (filtered)
            std_shap_filtered = filtered_shap_values.std(axis=0)
            top_var_count = min(10, len(filtered_feature_names))
            top_var_indices = np.argsort(std_shap_filtered)[-top_var_count:][::-1]
            axes[1, 1].barh(
                range(len(top_var_indices)),
                std_shap_filtered[top_var_indices],
                color="purple",
                alpha=0.7,
            )
            axes[1, 1].set_yticks(range(len(top_var_indices)))
            axes[1, 1].set_yticklabels(
                [filtered_feature_names[i] for i in top_var_indices], fontsize=9
            )
            axes[1, 1].set_xlabel("Std Dev of SHAP", fontsize=11)
            axes[1, 1].set_title("Top 10 Features by SHAP Variance", fontsize=12)
            axes[1, 1].grid(axis="x", alpha=0.3)

            plt.suptitle(
                f"SHAP Analysis Summary - {model_name} on {dataset_name}",
                fontsize=16,
                y=0.998,
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(save_dir, "shap_overall_summary.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
            logger.info("Overall summary plot saved")
        except Exception as e:
            logger.warning(f"Could not create overall summary plot: {e}")

        logger.info(f"All visualizations saved to {save_dir}")
