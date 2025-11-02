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
    Shapley explainer for temporal graph neural networks with proper coalition-based computation.
    
    Implements the Shapley value formula:
    φ_i(v) = Σ_{S⊆N\{i}} [|S|!(|N|-|S|-1)! / |N|!] * [v(S∪{i}) - v(S)]
    
    Players (Features):
    - Node features (source and destination)
    - Edge attributes/features  
    - Temporal information
    
    Note: Neighborhood structure is handled internally by the model and cannot be 
    independently controlled, so it's not included as a separate player.
    
    Args:
        model: Temporal GNN model with structure [temporal_encoder, link_predictor]
        node_raw_features: Full node feature matrix [num_nodes, node_feat_dim]
        edge_raw_features: Full edge feature matrix [num_edges, edge_feat_dim]
        device: Device for computation
        num_samples: Number of Monte Carlo samples for approximate Shapley
        use_sampling: Whether to use Monte Carlo approximation (True) or exact computation (False)
        batch_size: Size of batches to process
    """

    def __init__(
        self, 
        model: nn.Module,
        node_raw_features: np.ndarray,
        edge_raw_features: np.ndarray,
        device: str = "cpu", 
        num_samples: int = 10, 
        use_sampling: bool = False,
        batch_size: int = 32 # batch coalition
    ):
        warnings.filterwarnings("ignore")
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.model = model
        self.node_raw_features = node_raw_features
        self.edge_raw_features = edge_raw_features
        self.device = device
        self.num_samples = num_samples
        self.use_sampling = use_sampling
        self.batch_size = batch_size      
        
        # define player indices
        self.PLAYERS = {
            'node_features': 0,
            'edge_features': 1,
            'temporal_info': 2,
        }
        self.num_players = len(self.PLAYERS)

        self._compute_baselines()
        
        # pre-compute all coalitions for exact method
        if not use_sampling:
            self.all_coalitions = self._generate_all_coalitions()
            
        self.model.eval()  
            
    def compute_shapley_values(
        self,
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        labels: Optional[np.ndarray] = None,
        return_df: bool = True,
        process_batch_size: Optional[int] = None
    ) -> Dict:
        """
        Compute Shapley values with batched processing.
        
        Args:
            src_node_ids: Source node IDs
            dst_node_ids: Destination node IDs
            node_interact_times: Interaction timestamps
            labels: Ground truth labels
            return_df: Whether to return DataFrame
            process_batch_size: Size of batches to process (None = all at once)
            
        Returns:
            Dictionary with Shapley values
        """
        total_size = len(src_node_ids)
        
        # set baseline time
        self.baseline_time = np.min(node_interact_times) if len(node_interact_times) > 0 else 0.0
        
        # determine processing batch size
        if process_batch_size is None:
            # process in chunks to avoid memory issues
            if self.use_sampling:
                process_batch_size = min(1000, total_size)  # larger for sampling
            else:
                process_batch_size = min(500, total_size)  # smaller for exact
        
        # initialize storage
        all_shapley_values = {
            'node_features': np.zeros(total_size),
            'edge_features': np.zeros(total_size),
            'temporal_info': np.zeros(total_size),
        }
        
        self.logger.info(f"Computing Shapley values for {total_size} interactions...")
        self.logger.info(f"Method: {'Sampling' if self.use_sampling else 'Exact'}")
        self.logger.info(f"Processing batch size: {process_batch_size}")
        if self.use_sampling:
            self.logger.info(f"Samples per batch: {self.num_samples}")
        
        # process in batches
        num_batches = (total_size + process_batch_size - 1) // process_batch_size
        
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
                start_idx = batch_idx * process_batch_size
                end_idx = min(start_idx + process_batch_size, total_size)
                
                batch_src = src_node_ids[start_idx:end_idx]
                batch_dst = dst_node_ids[start_idx:end_idx]
                batch_times = node_interact_times[start_idx:end_idx]
                
                # compute Shapley values for this batch
                if self.use_sampling:
                    batch_shapley = self._compute_batch_sampling_shapley(
                        batch_src, batch_dst, batch_times
                    )
                else:
                    batch_shapley = self._compute_batch_exact_shapley(
                        batch_src, batch_dst, batch_times
                    )
                
                # store results
                for player, values in batch_shapley.items():
                    all_shapley_values[player][start_idx:end_idx] = values
        
        # restore original features
        self._restore_model_features()
        
        # compile results
        results = {
            'shapley_values': all_shapley_values,
            'src_node_ids': src_node_ids,
            'dst_node_ids': dst_node_ids,
            'node_interact_times': node_interact_times,
            'labels': labels
        }
        
        if return_df:
            results['shapley_df'] = self._format_results(results)
        
        return results
    
    def get_summary_statistics(self, results: Dict) -> pd.DataFrame:
        """
        Get summary statistics of Shapley values.

        Args:
            results (Dict): dictionary of Shapley values

        Returns:
            pd.DataFrame: DataFrame of summary statistics
        """
        shapley_values = results.get('shapley_values')
        if shapley_values is None:
            return None
        
        summary_data = []
        for feature, values in shapley_values.items():
            summary_data.append({
                'Feature': feature.replace('_', ' ').title(),
                'Mean_Shapley': np.mean(values),
                'Std_Shapley': np.std(values),
                'Min_Shapley': np.min(values),
                'Max_Shapley': np.max(values),
                'Median_Shapley': np.median(values),
                'Mean_Absolute_Shapley': np.mean(np.abs(values)),
                'Positive_Contrib_Pct': np.mean(values > 0) * 100,
            })
        
        return pd.DataFrame(summary_data)

    def get_feature_rankings(self, results: Dict) -> pd.DataFrame:
        """
        Rank features by importance.

        Args:
            results (Dict): dictionary of Shapley values

        Returns:
            pd.DataFrame: DataFrame of feature rankings
        """
        shapley_values = results.get('shapley_values')
        if shapley_values is None:
            return None
        
        rankings = []
        for feature, values in shapley_values.items():
            rankings.append({
                'Feature': feature.replace('_', ' ').title(),
                'Average_Importance': np.mean(np.abs(values)),
                'Average_Contribution': np.mean(values),
            })
        
        df = pd.DataFrame(rankings)
        df = df.sort_values('Average_Importance', ascending=False).reset_index(drop=True)
        df['Rank'] = range(1, len(df) + 1)
        
        return df[['Rank', 'Feature', 'Average_Importance', 'Average_Contribution']]

    def visualize_top_interactions(
        self, results: Dict, top_k: int = 10, sort_by: str = 'Temporal_Importance'
    ) -> pd.DataFrame:
        """
        Visualize top interactions by importance.

        Args:
            results (Dict): dictionary of Shapley values
            top_k (int, optional): number of top interactions to visualize. Defaults to 10.
            sort_by (str, optional): column to sort by. Defaults to 'Temporal_Importance'.

        Returns:
            pd.DataFrame: DataFrame of top interactions
        """
        df = results.get('shapley_df')
        if df is None:
            return None
        
        return df.nlargest(top_k, sort_by).reset_index(drop=True)
        
    def _compute_baselines(self):
        """
        Compute baseline/reference values for each feature type.
        """
        self.baselines = {}
        
        # node features: use mean
        if self.node_raw_features is not None and len(self.node_raw_features) > 0:
            self.baselines['node_features'] = np.mean(
                self.node_raw_features, axis=0, keepdims=True
            )
            self.baseline_node_feature_matrix = np.tile(
                self.baselines['node_features'], 
                (len(self.node_raw_features), 1)
            )
        else:
            self.baselines['node_features'] = None
            self.baseline_node_feature_matrix = None
            
        # edge features: use mean
        if self.edge_raw_features is not None and len(self.edge_raw_features) > 0:
            self.baselines['edge_features'] = np.mean(
                self.edge_raw_features, axis=0, keepdims=True
            )
            self.baseline_edge_feature_matrix = np.tile(
                self.baselines['edge_features'],
                (len(self.edge_raw_features), 1)
            )
        else:
            self.baselines['edge_features'] = None
            self.baseline_edge_feature_matrix = None
            
        self.baseline_time = None

    def _generate_all_coalitions(self) -> List[List[str]]:
        """
        Generate all possible coalitions for exact Shapley computation.
        
        Returns:
            List[List[str]]: List of all possible coalitions
        """
        player_names = list(self.PLAYERS.keys())
        all_coalitions = []
        
        for r in range(len(player_names) + 1):
            for coalition in combinations(player_names, r):
                all_coalitions.append(list(coalition))
        
        return all_coalitions

    def _set_model_features(self, coalition: List[str]):
        """
        Modify model's feature matrices based on coalition.
        
        Args:
            coalition (List[str]): list of features in coalition
        """
        # Node features
        if 'node_features' in coalition:
            self.model[0].node_raw_features = torch.from_numpy(
                self.node_raw_features
            ).float().to(self.device)
        else:
            if self.baseline_node_feature_matrix is not None:
                self.model[0].node_raw_features = torch.from_numpy(
                    self.baseline_node_feature_matrix
                ).float().to(self.device)
        
        # Edge features
        if 'edge_features' in coalition:
            self.model[0].edge_raw_features = torch.from_numpy(
                self.edge_raw_features
            ).float().to(self.device)
        else:
            if self.baseline_edge_feature_matrix is not None:
                self.model[0].edge_raw_features = torch.from_numpy(
                    self.baseline_edge_feature_matrix
                ).float().to(self.device)

    def _batch_get_predictions(
        self,
        src_ids: np.ndarray,
        dst_ids: np.ndarray,
        times: np.ndarray,
        coalition: List[str]
    ) -> torch.Tensor:
        """
        get predictions for a batch of interactions with given coalition.
        
        Args:
            src_ids: Array of source node IDs
            dst_ids: Array of destination node IDs
            times: Array of interaction times
            coalition: List of features in coalition
            
        Returns:
            Tensor of predictions
        """
        self._set_model_features(coalition)
        
        # use baseline time if temporal not in coalition
        actual_times = times if 'temporal_info' in coalition else np.full_like(times, self.baseline_time)
        
        # get embeddings for batch
        src_emb, dst_emb = self.model[0].compute_src_dst_node_temporal_embeddings(
            src_node_ids=src_ids,
            dst_node_ids=dst_ids,
            node_interact_times=actual_times
        )
        
        # get predictions
        preds = self.model[1](src_emb, dst_emb).squeeze().sigmoid()
        
        return preds

    def _compute_batch_exact_shapley(
        self,
        src_ids: np.ndarray,
        dst_ids: np.ndarray,
        times: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Compute exact Shapley values for a batch of interactions.
        
        Args:
            src_ids: Array of source node IDs
            dst_ids: Array of destination node IDs
            times: Array of interaction times
            
        Returns:
            Dictionary of Shapley values
        """
        batch_size = len(src_ids)
        player_names = list(self.PLAYERS.keys())
        n = len(player_names)
        
        # initialize Shapley values for batch
        shapley_values = {name: np.zeros(batch_size) for name in player_names}
        
        # cache predictions for all coalitions
        coalition_cache = {}
        
        self.logger.info(f"Computing predictions for {len(self.all_coalitions)} coalitions...")
        
        # compute predictions for all coalitions at once
        for coalition in self.all_coalitions:
            coalition_key = tuple(sorted(coalition))
            preds = self._batch_get_predictions(src_ids, dst_ids, times, coalition)
            coalition_cache[coalition_key] = preds.cpu().numpy()
        
        # compute Shapley values using cached predictions
        for i, player in enumerate(player_names):
            other_players = [p for j, p in enumerate(player_names) if j != i]
            
            for r in range(len(other_players) + 1):
                for subset in combinations(other_players, r):
                    coalition_without = tuple(sorted(subset))
                    coalition_with = tuple(sorted(list(subset) + [player]))
                    
                    # get cached predictions
                    pred_without = coalition_cache[coalition_without]
                    pred_with = coalition_cache[coalition_with]
                    
                    # marginal contribution for entire batch
                    marginal_contribution = pred_with - pred_without
                    
                    # shapley weight
                    s = len(subset)
                    weight = (math.factorial(s) * math.factorial(n - s - 1)) / math.factorial(n)
                    
                    shapley_values[player] += weight * marginal_contribution
        
        return shapley_values

    def _compute_batch_sampling_shapley(
        self,
        src_ids: np.ndarray,
        dst_ids: np.ndarray,
        times: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Compute approximate Shapley values for a batch using sampling.

        Args:
            src_ids: Array of source node IDs
            dst_ids: Array of destination node IDs
            times: Array of interaction times

        Returns:
            Dictionary of Shapley values
        """
        batch_size = len(src_ids)
        player_names = list(self.PLAYERS.keys())
        shapley_values = {name: np.zeros(batch_size) for name in player_names}
        
        for _ in range(self.num_samples):
            # random permutation
            permutation = np.random.permutation(player_names).tolist()
            
            coalition = []
            prev_preds = None
            
            for player in permutation:
                # get prediction without player
                if prev_preds is None:
                    prev_preds = self._batch_get_predictions(
                        src_ids, dst_ids, times, coalition
                    ).cpu().numpy()
                
                # add player
                coalition.append(player)
                
                # get prediction with player
                curr_preds = self._batch_get_predictions(
                    src_ids, dst_ids, times, coalition
                ).cpu().numpy()
                
                # marginal contribution
                marginal_contribution = curr_preds - prev_preds
                shapley_values[player] += marginal_contribution
                
                prev_preds = curr_preds
        
        # average over samples
        for player in player_names:
            shapley_values[player] /= self.num_samples
        
        return shapley_values

    def _restore_model_features(self):
        """
        Restore original features to model.
        """
        if self.node_raw_features is not None:
            self.model[0].node_raw_features = torch.from_numpy(
                self.node_raw_features
            ).float().to(self.device)
        
        if self.edge_raw_features is not None:
            self.model[0].edge_raw_features = torch.from_numpy(
                self.edge_raw_features
            ).float().to(self.device)

    def _format_results(self, results: Dict) -> pd.DataFrame:
        """
        Format Shapley values as DataFrame.
        
        Args:
            results: Dictionary of Shapley values
            
        Returns:
            DataFrame of Shapley values
        """
        shapley_values = results['shapley_values']
        
        df_data = {
            'Source_Node': results['src_node_ids'],
            'Destination_Node': results['dst_node_ids'],
            'Interaction_Time': results['node_interact_times'],
            'Node_Features_Shapley': shapley_values['node_features'],
            'Edge_Features_Shapley': shapley_values['edge_features'],
            'Temporal_Shapley': shapley_values['temporal_info'],
        }
        
        if results['labels'] is not None:
            df_data['Label'] = results['labels']
        
        df = pd.DataFrame(df_data)
        
        # Add importance metrics
        df['Node_Features_Importance'] = np.abs(shapley_values['node_features'])
        df['Edge_Features_Importance'] = np.abs(shapley_values['edge_features'])
        df['Temporal_Importance'] = np.abs(shapley_values['temporal_info'])
        
        # Dominant feature
        feature_cols = ['Node_Features_Shapley', 'Edge_Features_Shapley', 'Temporal_Shapley']
        df['Dominant_Feature'] = df[feature_cols].abs().idxmax(axis=1)
        df['Dominant_Feature'] = df['Dominant_Feature'].str.replace('_Shapley', '')
        
        return df
    