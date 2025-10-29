# load_model.py
# DyGLib-Explainability
# Justin Hoang
# 10/28/2025

import logging
import time
import os
import warnings
import torch.nn as nn

from models.TGAT import TGAT
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.CAWN import CAWN
from models.TCL import TCL
from models.GraphMixer import GraphMixer
from models.DyGFormer import DyGFormer
from models.modules import MergeLayer
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, get_neighbor_sampler, NegativeEdgeSampler
from evaluate_models_utils import (
    evaluate_edge_bank_link_prediction,
)
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


def load_link_prediction_model() -> nn.Module:
    """
    load the link prediction model

    Raises:
        ValueError: wrong value for model_name

    Returns:
        nn.Module: the link prediction model
    """
    # get the args for the link prediction task
    args = get_link_prediction_args(is_evaluation=True)
    
    # get the data for the link prediction task
    (
        node_raw_features,
        edge_raw_features,
        full_data,
        train_data,
        val_data,
        test_data,
        _,
        _,
    ) = get_link_prediction_data(
        dataset_name=args.dataset_name,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    
    # get the neighbor sampler for the link prediction task
    full_neighbor_sampler = get_neighbor_sampler(
        data=full_data,
        sample_neighbor_strategy=args.sample_neighbor_strategy,
        time_scaling_factor=args.time_scaling_factor,
        seed=1
    )
    
    # get the negative edge sampler for the link prediction task
    if args.negative_sample_strategy != "random":
        test_neg_edge_sampler = NegativeEdgeSampler(
            src_node_ids=full_data.src_node_ids,
            dst_node_ids=full_data.dst_node_ids,
            interact_times=full_data.node_interact_times,
            last_observed_time=val_data.node_interact_times[-1],
            negative_sample_strategy=args.negative_sample_strategy,
            seed=2,
        )
    else:
        test_neg_edge_sampler = NegativeEdgeSampler(
            src_node_ids=full_data.src_node_ids,
            dst_node_ids=full_data.dst_node_ids,
            seed=2,
        )
        
    # get the data loader for the link prediction task
    test_idx_data_loader = get_idx_data_loader(
        indices_list=list(range(len(test_data.src_node_ids))),
        batch_size=args.batch_size,
        shuffle=False,
    )
    
    # evaluate the link prediction model
    # we separately evaluate EdgeBank, since EdgeBank does not contain any trainable parameters and has a different evaluation pipeline
    if args.model_name == "EdgeBank":
        evaluate_edge_bank_link_prediction(
            args=args,
            train_data=train_data,
            val_data=val_data,
            test_idx_data_loader=test_idx_data_loader,
            test_neg_edge_sampler=test_neg_edge_sampler,
            test_data=test_data,
        )
    else:
        # evaluate the link prediction model for multiple runs
        for run in range(args.num_runs):
            set_random_seed(seed=run)
            
            args.seed = run
            args.load_model_name = f"{args.model_name}_seed{args.seed}"
            args.save_result_name = f"{args.negative_sample_strategy}_negative_sampling_{args.model_name}_seed{args.seed}"
            
            # create the folder for the logs
            os.makedirs(
                f"./logs/{args.model_name}/{args.dataset_name}/{args.save_result_name}/",
                exist_ok=True
            )
            
            # create the file handler for the logs
            fh = logging.FileHandler(
                f"./logs/{args.model_name}/{args.dataset_name}/{args.save_result_name}/{str(time.time())}.log"
            )
            fh.setLevel(logging.DEBUG)
            
            # create the stream handler for the logs
            ch = logging.StreamHandler()
            ch.setLevel(logging.WARNING)
            
            # create the formatter for the logs
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            # set the formatter for the file handler
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            
            # add the file handler and the stream handler to the logger
            logger.addHandler(fh)
            logger.addHandler(ch)
            
            # start the time for the run
            run_start_time = time.time()
            logger.info(f"********** Run {run + 1} starts. **********")
            
            # log the configuration
            logger.info(f"configuration is {args}")
            
            # create model
            if args.model_name == "TGAT":
                dynamic_backbone = TGAT(
                    node_raw_features=node_raw_features,
                    edge_raw_features=edge_raw_features,
                    neighbor_sampler=full_neighbor_sampler,
                    time_feat_dim=args.time_feat_dim,
                    num_layers=args.num_layers,
                    num_heads=args.num_heads,
                    dropout=args.dropout,
                    device=args.device,
                )
            elif args.model_name in ["JODIE", "DyRep", "TGN"]:
                # four floats that represent the mean and standard deviation of source and destination node time shifts in the training data, which is used for JODIE
                (
                    src_node_mean_time_shift,
                    src_node_std_time_shift,
                    dst_node_mean_time_shift_dst,
                    dst_node_std_time_shift,
                ) = compute_src_dst_node_time_shifts(
                    train_data.src_node_ids,
                    train_data.dst_node_ids,
                    train_data.node_interact_times
                )
                dynamic_backbone = MemoryModel(
                    node_raw_features=node_raw_features,
                    edge_raw_features=edge_raw_features,
                    neighbor_sampler=full_neighbor_sampler,
                    time_feat_dim=args.time_feat_dim,
                    model_name=args.model_name,
                    num_layers=args.num_layers,
                    num_heads=args.num_heads,
                    dropout=args.dropout,
                    src_node_mean_time_shift=src_node_mean_time_shift,
                    src_node_std_time_shift=src_node_std_time_shift,
                    dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst,
                    dst_node_std_time_shift=dst_node_std_time_shift,
                    device=args.device,
                )
            elif args.model_name == "CAWN":
                dynamic_backbone = CAWN(
                    node_raw_features=node_raw_features,
                    edge_raw_features=edge_raw_features,
                    neighbor_sampler=full_neighbor_sampler,
                    time_feat_dim=args.time_feat_dim,
                    position_feat_dim=args.position_feat_dim,
                    walk_length=args.walk_length,
                    num_walk_heads=args.num_walk_heads,
                    dropout=args.dropout,
                    device=args.device,
                )
            elif args.model_name == "TCL":
                dynamic_backbone = TCL(
                    node_raw_features=node_raw_features,
                    edge_raw_features=edge_raw_features,
                    neighbor_sampler=full_neighbor_sampler,
                    time_feat_dim=args.time_feat_dim,
                    num_layers=args.num_layers,
                    num_heads=args.num_heads,
                    num_depths=args.num_neighbors + 1,
                    dropout=args.dropout,
                    device=args.device,
                )
            elif args.model_name == "GraphMixer":
                dynamic_backbone = GraphMixer(
                    node_raw_features=node_raw_features,
                    edge_raw_features=edge_raw_features,
                    neighbor_sampler=full_neighbor_sampler,
                    time_feat_dim=args.time_feat_dim,
                    num_tokens=args.num_neighbors,
                    num_layers=args.num_layers,
                    dropout=args.dropout,
                    device=args.device,
                )
            elif args.model_name == "DyGFormer":
                dynamic_backbone = DyGFormer(
                    node_raw_features=node_raw_features,
                    edge_raw_features=edge_raw_features,
                    neighbor_sampler=full_neighbor_sampler,
                    time_feat_dim=args.time_feat_dim,
                    channel_embedding_dim=args.channel_embedding_dim,
                    patch_size=args.patch_size,
                    num_layers=args.num_layers,
                    num_heads=args.num_heads,
                    dropout=args.dropout,
                    max_input_sequence_length=args.max_input_sequence_length,
                    device=args.device,
                )
            else:
                raise ValueError(f"Wrong value for model_name {args.model_name}!")
            
            link_predictor = MergeLayer(
                input_dim1=node_raw_features.shape[1],
                input_dim2=node_raw_features.shape[1],
                hidden_dim=node_raw_features.shape[1],
                output_dim=1,
            )
            
            model = nn.Sequential(dynamic_backbone, link_predictor)
            
            logger.info(f"model -> {model}")
            logger.info(
                f"model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, "
                f"{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB."
            )
            
            # load the saved model
            load_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.load_model_name}"
            early_stopping = EarlyStopping(
                patience=0,
                save_model_folder=load_model_folder,
                save_model_name=args.load_model_name,
                logger=logger,
                model_name=args.model_name
            )
            early_stopping.load_checkpoint(model, map_location="cpu")
            
            model = convert_to_gpu(model, device=args.device)
            
    return model

def load_node_prediction_model() -> nn.Module:
    # TODO: implement
    pass


# TODO: remove
if __name__ == "__main__":
    model = load_link_prediction_model()
    
    print(model)