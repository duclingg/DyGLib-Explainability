# DyGFormer Explain

This module attempts to gain insight on AI Explanations for temporal graph neural networks, based on the framework `TGB` (Temporal Graph Benchmark), and built off of the research paper `DyGFormer`.  

## How to Run
In order to get an evaluation from a graph neural network model, you must first train it.  

Train your desired model (check root `README` for reference) and prediction method (dynamic link prediction or dynamic node classification).  

When running your desired explanation, you must make sure you pass in the same arguments as when you trained it. 
Make sure at least one fully trained pickle `.pkl` file exists for explanation.

### Dynamic Link-Prediction Models
Here is an example of how to run a SHAP-value explanation on a dynamic link prediction model, `DyGFormer`, on the dataset `wikipedia`.  
This model was trained on the following parameters:  
- `patch_size 2`
- `max_input_sequence 64`
- `negative_sample_strategy random`
- `num_runs 5`
- `gpu 0`  
You must define the explanation type, as well as the prediction task.

```zsh
python explain_model.py --dataset_name wikipedia --model_name DyGFormer --patch_size 2 --max_input_sequence 64 --negative_sample_strategy random --num_runs 5 --gpu 0 --explanation_type shapley --prediction_type link
```