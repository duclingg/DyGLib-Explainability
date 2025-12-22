# utils.py
# DyGLib-Explainability
# Justin Hoang
# 12/21/2025

import os
import random
import logging
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict
from typing import List, Optional, Dict, Any, Tuple

from sklearn.metrics import roc_auc_score, average_precision_score

from tqdm import tqdm

from utils.DataLoader import Data

from explain.TempME.utils import NeighborFinder


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def compute_teacher_predictions(
    model: nn.Module, data: Data, indices: np.ndarray
) -> List[Data]:
    example_data = []

    logger.info(f"Computing teacher predictions for {len(indices)} examples...")
    for i in tqdm(indices):
        src = data.src_node_ids[i]
        dst = data.dst_node_ids[i]
        ts = data.node_interact_times[i]
        edge_id = data.edge_ids[i]
        
        model.eval()
        with torch.no_grad():
            src_emb, dst_emb = model[0].compute_src_dst_node_temporal_embeddings(
                np.array([src]), np.array([dst]), np.array([ts])
            )
            teacher_logits = model[1](src_emb, dst_emb)
            teacher_prob = teacher_logits.sigmoid().item()
            teacher_label = 1.0 if teacher_prob > 0.5 else 0.0
            
        example_data.append({
            "src": src,
            "dst": dst,
            "ts": ts,
            "edge_id": edge_id,
            "teacher_logit": teacher_logits.item(),
            "teacher_label": teacher_label
        })
        
    logger.info(f"""
        Teacher labels: {sum([e['teacher_label'] for e in example_data])} postive, 
        {len(example_data) - sum([e['teacher_label'] for e in example_data])} negative
    """)
    
    return example_data

def extract_walks(
    src_id: int, 
    interact_time: float, 
    neighbor_finder: NeighborFinder, 
    num_walks: int
):
    src_idx_l = np.array([src_id])
    cut_time_l = np.array([interact_time])
    
    x, y, z = neighbor_finder.get_temporal_neighbor(
        src_idx_l=src_idx_l,
        cut_time_l=cut_time_l,
        num_neighbor=num_walks,
    )
    subgraph_src = ([x], [y], [z])
    
    node_records, edge_idx_records, ts_records, out_anony = neighbor_finder.find_k_walks(
        degree=num_walks,
        src_idx_l=src_idx_l,
        num_neighbors=num_walks,
        subgraph_src=subgraph_src
    )
    
    cat_feat = out_anony[:, :, 0:1]
    
    return (node_records, edge_idx_records, ts_records, cat_feat, None)

def train_tempme(
    data: List[Data], 
    batch_size: int,
    num_walks: int,
    prior_p: float,
    beta: float,
    neighbor_finder: NeighborFinder,
    tempme: nn.Module, 
    optimizer: torch.optim.Adam,
    criterion: nn.BCEWithLogitsLoss,
    device: torch.device,
    epochs: int = 50
) -> None:
    train_losses = []
    rec_losses = []
    kl_losses = []
    
    for epoch in tqdm(range(epochs), desc="Training TempME"):
        epoch_loss = 0.0
        epoch_rec_loss = 0.0
        epoch_kl_loss = 0.0
        
        shuffled_indcies = np.random.permutation(len(data))
        shuffled_examples = [data[i] for i in shuffled_indcies]
        
        for batch_start in range(0, len(shuffled_examples), batch_size):
            batch_end = min(batch_start + batch_size, len(shuffled_examples))
            batch_examples = shuffled_examples[batch_start:batch_end]
            
            optimizer.zero_grad()
            
            batch_logits = []
            batch_labels = []
            batch_kl_losses = []
            
            for example in batch_examples:
                src = example['src']
                dst = example['dst']
                ts = example['ts']
                edge_id = example['edge_id']
                teacher_label = example['teacher_label']
                
                walks = extract_walks(src, ts, neighbor_finder, num_walks)
                _, edge_idx_walks, _, _, _ = walks
                
                edge_identify = (edge_idx_walks == edge_id).astype(int)
                
                graphlet_imp = tempme(walks, [ts], edge_identify)
                
                surrogate_logit = torch.logit(torch.clamp(graphlet_imp.mean(), 1e-7, 1-1e-7))  # convert to logit
                
                batch_logits.append(surrogate_logit)
                batch_labels.append(teacher_label)
                
                # kl loss with prior_p
                kl_loss = tempme.kl_loss(graphlet_imp, walks, target=prior_p)
                batch_kl_losses.append(kl_loss)
                
            # stack predictions and labels
            pred = torch.stack(batch_logits).unsqueeze(1)  # [batch_size, 1]
            y_ori = torch.tensor(batch_labels, device=device).float().unsqueeze(1)  # [batch_size, 1]
            
            # binary cross-entropy loss using the criterion
            pred_loss = criterion(pred, y_ori)
            
            # total KL loss
            kl_loss_total = sum(batch_kl_losses) / len(batch_kl_losses)
            
            # combined loss
            loss = pred_loss + beta * kl_loss_total
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_rec_loss += pred_loss.item()
            epoch_kl_loss += kl_loss_total.item()
        
        num_batches = (len(data) + batch_size - 1)
        epoch_loss /= num_batches
        epoch_rec_loss /= num_batches
        epoch_kl_loss /= num_batches
        
        train_losses.append(epoch_loss)
        rec_losses.append(epoch_rec_loss)
        kl_losses.append(epoch_kl_loss)
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: Loss {epoch_loss:.4f} (Rec: {epoch_rec_loss:.4f}, KL: {epoch_kl_loss:.4f})")

    logger.info(f"Training completed! Final loss: {train_losses[-1]:.4f}")
    return None

def get_explanation_parameters(
    test_data: Data,
    tempme: nn.Module, 
    neighbor_finder: NeighborFinder, 
    num_walks: int,
    src: int, 
    dst: int, 
    ts: float, 
    edge_id: int, 
    bgd: Optional[int] = None,
) -> Dict[str, Any]:
    cut_time_l = np.array([ts])
    
    src_idx_l = np.array([src])
    subgraph_src = neighbor_finder.find_k_hop(
        k=2,
        src_idx_l=src_idx_l,
        cut_time_l=cut_time_l,
        num_neighbors=num_walks,
        e_idx_l=np.array([edge_id])
    )
    
    dst_idx_l = np.array([dst])
    subgraph_tgt = neighbor_finder.find_k_hop(
        k=2,
        src_idx_l=dst_idx_l,
        cut_time_l=cut_time_l,
        num_neighbors=num_walks,
        e_idx_l=np.array([edge_id])
    )
    
    if bgd is None:
        nodes_before_time = set()
        for i in range(len(test_data.src_node_ids)):
            if test_data.node_interact_times[i] < ts:
                nodes_before_time.add(test_data.src_node_ids[i])
                nodes_before_time.add(test_data.dst_node_ids[i])
        
        candidate_nodes = [n for n in nodes_before_time if n != src and n != dst]
        
        # fallback
        if len(candidate_nodes) == 0:
            candidate_nodes = [n for n in range(test_data.num_unique_nodes) if n != src and n != dst]
        
        if len(candidate_nodes) == 0:
            raise ValueError("No valid background nodes available")
        
        bgd = random.choice(candidate_nodes)

    bgd_idx_l = np.array([bgd])
    subgraph_bgd = neighbor_finder.find_k_hop(
        k=2,
        src_idx_l=bgd_idx_l,
        cut_time_l=cut_time_l,
        num_neighbors=num_walks,
        e_idx_l=None
    )
    
    walks_src = extract_walks(src, ts, neighbor_finder, num_walks)
    walks_tgt = extract_walks(dst, ts, neighbor_finder, num_walks)
    walks_bgd = extract_walks(bgd, ts, neighbor_finder, num_walks)
    
    tempme.eval()
    with torch.no_grad():
        edge_identify_src = (walks_src[1] == edge_id).astype(int)
        graphlet_imp_src = tempme(walks_src, [ts], edge_identify_src)
        
        edge_identify_tgt = (walks_tgt[1] == edge_id).astype(int)
        graphlet_imp_tgt = tempme(walks_tgt, [ts], edge_identify_tgt)
        
        edge_identify_bgd = (walks_bgd[1] == edge_id).astype(int)
        graphlet_imp_bgd = tempme(walks_bgd, [ts], edge_identify_bgd)
        
    return {
        "subgraph_src": subgraph_src,
        "graphlet_imp_src": graphlet_imp_src,
        "walks_src": walks_src,
        "subgraph_tgt": subgraph_tgt,
        "graphlet_imp_tgt": graphlet_imp_tgt,
        "walks_tgt": walks_tgt,
        "subgraph_bgd": subgraph_bgd,
        "graphlet_imp_bgd": graphlet_imp_bgd,
        "walks_bgd": walks_bgd,
        "training": False
    }
    
def get_explanations(
    model_name: str,
    dataset_name: str,
    test_data: Data,
    examples: List[Data],
    tempme: nn.Module,
    neighbor_finder: NeighborFinder,
    num_walks: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info(f"Getting explanations for {len(examples)} examples...")
    
    motif_subgraphs = defaultdict(list)
    graphlet_imp = defaultdict(list)
    edge_imp_scores = defaultdict(list)
    
    for example in tqdm(examples):
        params = get_explanation_parameters(
            test_data=test_data,
            tempme=tempme,
            neighbor_finder=neighbor_finder,
            num_walks=num_walks,
            src=example['src'],
            dst=example['dst'],
            ts=example['ts'],
            edge_id=example['edge_id'],
        )
        
        edge_imp_result = tempme.retrieve_explanation(**params)[0]
        
        # Get lengths to split the concatenated tensor back into src, tgt, bgd parts
        src_len = params['graphlet_imp_src'].shape[0]
        tgt_len = params['graphlet_imp_tgt'].shape[0]
        bgd_len = params['graphlet_imp_bgd'].shape[0]
        
        # Split the concatenated tensor: [src, tgt, bgd]
        edge_imp_src = edge_imp_result[:src_len]
        edge_imp_tgt = edge_imp_result[src_len:src_len+tgt_len]
        edge_imp_bgd = edge_imp_result[src_len+tgt_len:src_len+tgt_len+bgd_len]
        
        motif_subgraphs['src'].append(params['subgraph_src'])
        motif_subgraphs['dst'].append(params['subgraph_tgt'])
        motif_subgraphs['bgd'].append(params['subgraph_bgd'])
        
        graphlet_imp['src'].append(params['graphlet_imp_src'].mean().item())
        graphlet_imp['dst'].append(params['graphlet_imp_tgt'].mean().item())
        graphlet_imp['bgd'].append(params['graphlet_imp_bgd'].mean().item())
        
        edge_imp_scores['src'].append(edge_imp_src.mean().item())
        edge_imp_scores['dst'].append(edge_imp_tgt.mean().item())
        edge_imp_scores['bgd'].append(edge_imp_bgd.mean().item())
        
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    path = os.path.join(project_root, 'saved_explanations', model_name, dataset_name, 'tempme')
    os.makedirs(path, exist_ok=True)
    
    # Create plots directory
    plots_dir = os.path.join(path, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    motif_subgraphs = pd.DataFrame(motif_subgraphs)
    motif_subgraphs.to_csv(os.path.join(path, 'motif_subgraphs.csv'), index=False)
    logger.info(f"Saved motif subgraphs to {os.path.join(path, 'motif_subgraphs.csv')}")
    
    graphlet_imp = pd.DataFrame(graphlet_imp)
    graphlet_imp.to_csv(os.path.join(path, 'graphlet_imp.csv'), index=False)
    logger.info(f"Saved graphlet importance to {os.path.join(path, 'graphlet_imp.csv')}")
    
    edge_imp_scores = pd.DataFrame(edge_imp_scores)
    edge_imp_scores.to_csv(os.path.join(path, 'edge_imp.csv'), index=False)
    logger.info(f"Saved edge importance to {os.path.join(path, 'edge_imp.csv')}")
    
    # distribution plots
    logger.info("Creating plots for graphlet importance...")
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Graphlet Importance Distributions - {model_name} on {dataset_name}', fontsize=16, fontweight='bold')
        
        for idx, key in enumerate(['src', 'dst', 'bgd']):
            axes[idx].hist(graphlet_imp[key], bins=50, alpha=0.7, color=['blue', 'green', 'red'][idx])
            axes[idx].set_xlabel('Importance Score')
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_title(f'{key.upper()} Distribution')
            axes[idx].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'graphlet_imp_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved graphlet importance plots to {os.path.join(plots_dir, 'graphlet_imp_plot.png')}")
    except Exception as e:
        logger.warning(f"Could not create graphlet importance plots: {e}")
    
    logger.info("Creating plots for edge importance...")
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Edge Importance Distributions - {model_name} on {dataset_name}', fontsize=16, fontweight='bold')
        
        for idx, key in enumerate(['src', 'dst', 'bgd']):
            axes[idx].hist(edge_imp_scores[key], bins=50, alpha=0.7, color=['blue', 'green', 'red'][idx])
            axes[idx].set_xlabel('Importance Score')
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_title(f'{key.upper()} Distribution')
            axes[idx].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'edge_imp_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved edge importance plots to {os.path.join(plots_dir, 'edge_imp_plot.png')}")
    except Exception as e:
        logger.warning(f"Could not create edge importance plots: {e}")
    
    logger.info(f"All plots saved to {plots_dir}")
    
    return motif_subgraphs, graphlet_imp, edge_imp_scores

def assess_explanations(
    test_data: Data,
    examples: List[Data],
    model: nn.Module,
    tempme: nn.Module,
    neighbor_finder: NeighborFinder,
    num_walks: int,
) -> Tuple[float, float, float, float, float, float, float, float]:
    logger.info(f"Assessing explanations for {len(examples)} examples...")
    base_probs, base_logits = [], []
    expl_probs, expl_logits = [], []
    true_labels = []
    
    for i, example in enumerate(tqdm(examples)):
        src = example['src']
        dst = example['dst']
        ts = example['ts']
        edge_id = example['edge_id']
        teacher_label = example['teacher_label']
        
        true_labels.append(teacher_label)
        
        # base model prediction
        model.eval()
        with torch.no_grad():
            src_emb_full, dst_emb_full = model[0].compute_src_dst_node_temporal_embeddings(
                np.array([src]), np.array([dst]), np.array([ts])
            )
            base_logit = model[1](src_emb_full, dst_emb_full)
            base_prob = base_logit.sigmoid().item()
            
            base_logits.append(base_logit.item())
            base_probs.append(base_prob)
            
        params = get_explanation_parameters(
            test_data=test_data,
            tempme=tempme,
            neighbor_finder=neighbor_finder,
            num_walks=num_walks,
            src=src,
            dst=dst,
            ts=ts,
            edge_id=edge_id,
        )
        
        graphlet_imp_src_prob = params['graphlet_imp_src'].mean().item()
        graphlet_imp_tgt_prob = params['graphlet_imp_tgt'].mean().item()
        
        tempme.eval()
        with torch.no_grad():
            graphlet_imp_src_logit = torch.logit(
                torch.clamp(torch.tensor(graphlet_imp_src_prob), 1e-7, 1-1e-7)
            )
            graphlet_imp_tgt_logit = torch.logit(
                torch.clamp(torch.tensor(graphlet_imp_tgt_prob), 1e-7, 1-1e-7)
            )
            weighted_logit = (graphlet_imp_src_logit + graphlet_imp_tgt_logit) / 2.0
            weighted_prob = torch.sigmoid(weighted_logit)
            
            expl_logits.append(weighted_logit.item())
            expl_probs.append(weighted_prob.item())
            
    base_probs = np.array(base_probs)
    base_logits = np.array(base_logits)
    expl_probs = np.array(expl_probs)
    expl_logits = np.array(expl_logits)
    true_labels = np.array(true_labels)
    
    # fidelity probability
    fid_prob = 1.0 - np.mean(np.abs(expl_probs - base_probs))
    
    # normalize logits before computing MAE (z-score normalization)
    logit_mean = np.mean(base_logits)
    logit_std = np.std(base_logits) + 1e-7
    
    norm_base_logits = (base_logits - logit_mean) / logit_std
    norm_expl_logits = (expl_logits - logit_mean) / logit_std
    
    # compute MAE and fidliety logit
    norm_mae = np.mean(np.abs(norm_expl_logits - norm_base_logits))
    fid_logit = 1.0 / (1.0 + norm_mae)
    
    expl_auc = roc_auc_score(true_labels, expl_probs)
    expl_ap = average_precision_score(true_labels, expl_probs)
    expl_acc = np.mean((expl_probs > 0.5) == true_labels)
    
    base_auc = roc_auc_score(true_labels, base_probs)
    base_ap = average_precision_score(true_labels, base_probs)
    base_acc = np.mean((base_probs > 0.5) == true_labels)
    
    print("\n" + "="*80)
    print("FIDELITY EVALUATION")
    print("="*80)

    print("\nBase Model (No Explanation):")
    print(f"  APS: {base_ap:.4f}")
    print(f"  AUC: {base_auc:.4f}")
    print(f"  ACC: {base_acc:.4f}")

    print("\nWith Explanations:")
    print(f"  APS: {expl_ap:.4f}")
    print(f"  AUC: {expl_auc:.4f}")
    print(f"  ACC: {expl_acc:.4f}")

    print("\nFidelity Metrics:")
    print(f"  Fidelity Prob:  {fid_prob:.4f}")
    print(f"  Fidelity Logit: {fid_logit:.4f}")

    print("\n" + "="*80)
    print("INTERPRETATION:")
    print("="*80)
    print("Fidelity measures how much the explanation changes predictions")
    print("  Higher fidelity = explanation preserves original model behavior")
    print("  Lower fidelity = explanation significantly alters predictions")

    print(f"\nFidelity prob: {fid_prob:.4f}")
    if fid_prob > 0.85:
        print("  High: Explanations preserve model predictions well")
    elif fid_prob > 0.65:
        print("  Moderate: Some prediction changes with explanations")
    else:
        print("  Low: Explanations significantly change predictions")
        
    print(f"\nFidelity logit: {fid_logit:.4f}")
    if fid_logit > 0.5:
        print("  High: Explanations preserve model predictions well")
    elif fid_logit > 0.3:
        print("  Moderate: Some prediction changes with explanations")
    else:
        print("  Low: Explanations significantly change predictions")
        
    return fid_prob, fid_logit, expl_auc, expl_ap, expl_acc, base_auc, base_ap, base_acc