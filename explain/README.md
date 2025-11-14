# DyGFormer Explain

This module provides AI Explainability tools for temporal graph neural networks, based on the framework `TGB` (Temporal Graph Benchmark), and built off of the research paper `DyGFormer`.

## Overview

The explainer module currently supports **SHAP (SHapley Additive exPlanations)** for explaining link predictions in temporal graphs. The implementation uses **KernelSHAP** with efficient sampling to handle large temporal graph datasets.

## Features

### SHAP Explainer
- **KernelSHAP Implementation**: Uses kernel-based approximation for computing Shapley values
- **Efficient Sampling**: Automatically samples a subset of interactions for explanation (default: 10%)
- **Background Data**: Uses historical interactions as background for SHAP computation
- **Feature Engineering**: Creates interpretable features from temporal graph data including:
  - Source node features
  - Destination node features
  - Edge features (if available)
  - Temporal information (timestamps)
  - Normalized node IDs

### Visualizations
The explainer generates comprehensive visualizations including:

1. **Beeswarm Plot** - Shows the distribution of SHAP values for all features
2. **Bar Plot** - Displays mean absolute SHAP values for feature importance ranking
3. **Waterfall Plots** - Individual prediction explanations (first 5 samples)
4. **Heatmap** - SHAP values across samples and features
5. **Feature Importance Plot** - Top features by SHAP importance
6. **Overall Summary** - Statistical analysis including:
   - Distribution of SHAP values
   - Top 10 features by mean |SHAP|
   - Positive vs negative contributions
   - Feature variance analysis

### Results Output
- **JSON File** - Contains:
  - SHAP values for all explained samples
  - Feature names and importance scores
  - Base value (expected model output)
  - Top 10 most important features
  - Metadata (number of samples, features, etc.)

## How to Run

### Prerequisites
1. **Train a Model First**: You must first train a temporal GNN model using the training scripts in the root directory
2. **Ensure Model Exists**: At least one fully trained pickle `.pkl` file must exist in `./saved_models/`
3. **Match Training Arguments**: Use the same arguments as when you trained the model

### Dynamic Link Prediction Explanation

Here is an example of how to run SHAP explanation on a dynamic link prediction model (`DyGFormer`) on the `wikipedia` dataset.

**Run Explanation:**
Example:  
```bash
python explain_model.py \
  --dataset_name wikipedia \
  --model_name DyGFormer \
  --patch_size 2 \
  --max_input_sequence_length 64 \
  --negative_sample_strategy random \
  --num_runs 5 \
  --gpu 0 \
  --explainer_type shapley \
  --prediction_type link
```

### Additional Examples

**Explain TGAT model on mooc dataset:**
Example:  
```bash
python explain_model.py \
  --dataset_name mooc \
  --model_name TGAT \
  --explainer_type shapley \
  --prediction_type link \
  --gpu 0
```

**Explain GraphMixer on reddit dataset:**
Example:  
```bash
python explain_model.py \
  --dataset_name reddit \
  --model_name GraphMixer \
  --explainer_type shapley \
  --prediction_type link \
  --num_runs 5 \
  --gpu 0
```

## Output Structure

After running the explainer, results will be saved in the following structure:

```
saved_explanations/
└── {model_name}/
    └── {dataset_name}/
        └── shapley/
            ├── shap_results.json           # SHAP values and feature importance
            └── plots/
                ├── shap_beeswarm.png       # Overall feature importance
                ├── shap_bar.png            # Feature importance bar chart
                ├── shap_heatmap.png        # SHAP values heatmap
                ├── shap_feature_importance.png  # Top features
                ├── shap_overall_summary.png     # Statistical summary
                ├── shap_waterfall_sample_1.png  # Individual explanation 1
                ├── shap_waterfall_sample_2.png  # Individual explanation 2
                ├── shap_waterfall_sample_3.png  # Individual explanation 3
                ├── shap_waterfall_sample_4.png  # Individual explanation 4
                └── shap_waterfall_sample_5.png  # Individual explanation 5
```

### Example Output (shap_results.json)

```json
{
  "base_value": 0.4523,
  "num_features": 10,
  "num_samples_explained": 100,
  "background_size": 100,
  "top_10_features": {
    "src_node_feat_0": 0.1234,
    "dst_node_feat_0": 0.0987,
    "timestamp": 0.0876,
    ...
  },
  "feature_importance": {
    "src_node_feat_0": 0.1234,
    ...
  },
  "shap_values": [...],
  "feature_names": [...],
  "sampled_indices": [...]
}
```

## Configuration Options

### Sampling Parameters
You can modify sampling parameters in `explain/shapley.py`:

```python
shapley_explainer = ShapleyExplainer(
    model=model,
    node_raw_features=node_raw_features,
    edge_raw_features=edge_raw_features,
    num_samples=100,      # Background samples for KernelSHAP
    sample_ratio=0.1,     # Ratio of test data to explain (10%)
    device=args.device,
)
```

- **num_samples**: Number of background samples used by KernelSHAP (more = more accurate but slower)
- **sample_ratio**: Fraction of test interactions to explain (lower = faster but less coverage)

### SHAP Computation Parameters
In the `compute_shapley_values` method:

```python
self.shap_values = self.shap_explainer.shap_values(
    explain_data,
    nsamples=100,     # Number of model evaluations per sample
    l1_reg="aic",     # Regularization for feature selection
)
```

## Implementation Details

### Feature Construction
The explainer creates a feature matrix from temporal graph interactions:

1. **Node Features**: Raw features for source and destination nodes
2. **Edge Features**: Raw edge features (if available)
3. **Temporal Features**: Interaction timestamps
4. **Structural Features**: Normalized node IDs

### Model Wrapping
The explainer wraps the temporal GNN model to make it compatible with SHAP:
- Handles the sequential model structure (backbone + predictor)
- Manages temporal embeddings computation
- Converts graph data to feature matrix format

## Interpreting Results

### Feature Importance
- **High |SHAP| Value**: Feature has strong influence on predictions
- **Positive SHAP**: Feature pushes prediction higher
- **Negative SHAP**: Feature pushes prediction lower

### Waterfall Plots
- Shows how each feature contributes to a single prediction
- Starts from base value (expected output)
- Each bar shows the contribution of one feature
- Final value is the model's prediction

### Beeswarm Plot
- Each point is one sample
- Horizontal position = SHAP value (contribution)
- Vertical position = feature
- Color = feature value (high/low)