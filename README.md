# DyGFormer Explainability

This projects implements T-GNN explainability methods, based on the framework [TGB](https://tgb.complexdatalab.com) (Temporal Graph Benchmark), and built off of the research paper [DyGFormer](https://github.com/yule-BUAA/DyGLib).  

Can be ran on the following explanation methods:  
- [TempME](https://github.com/Graph-and-Geometric-Learning/TempME) - Motif Discovery
- SHAP - Feature-based importance (game theory)
  - WARNGING: Not assessed for faithfulness/fidelity

### Prerequisites
1. **Train a Model First**: You must first train a temporal GNN model using the training scripts in the root directory
2. **Ensure Model Exists**: At least one fully trained pickle `.pkl` file must exist in `./saved_models/`
3. **Match Training Arguments**: Use the same arguments as when you trained the model

### Dynamic Link Prediction Explanation

Here is an example of how to run SHAP explanation on a dynamic link prediction model (`DyGFormer`) on the `wikipedia` dataset.

**Preprocess Data:**  
Example:
```bash
cd precprocess_data
python preprocess_data.py \
  --dataset_name wikipedia
```

**Train TGNN Model:**  
Example:  
```bash
python train_link_prediction.py \
  --dataset_name wikipedia \
  --model_name DyGFormer \
  --patch_size 2 \
  --max_input_sequence_length 64 \
  --negative_sample_strategy random \
  --num_runs 5 \
  --gpu 0
```  

On best model config:  
```bash
python train_link_prediction.py \
  --dataset_name wikipedia \
  --model_name DyGFormer \
  --load_best_configs \
  --num_runs 5 \
  --gpu 0
```

**Evaluate Model:**  
Example:
```bash
python evaluate_link_prediction.py \
  --dataset_name wikipedia \
  --model_name DyGFormer \
  --patch_size 2 \
  --max_input_sequence_length 64 \
  --negative_sample_strategy random \
  --num_runs 5 \
  --gpu 0
```

On best model config:  
```bash
python evaluate_link_prediction.py \
  --dataset_name wikipedia \
  --model_name DyGFormer \
  --load_best_configs \
  --num_runs 5 \
  --gpu 0
```

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
  --explainer_type tempe \
  --prediction_type link
```

On best model config:  
```bash
python explain_model.py \
  --dataset_name wikipedia \
  --model_name DyGFormer \
  --load_best_configs \
  --num_runs 5 \
  --gpu 0 \
  --explainer_type tempme \
  --prediction_type link
```

Thanks to [@yule-BUAA]() for DyGFormer and [@Cather-Chen](https://github.com/Cather-Chen) for TempME