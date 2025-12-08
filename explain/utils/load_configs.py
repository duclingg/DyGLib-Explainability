# load_configs.py
# DyGLib-Explainability
# Justin Hoang
# 10/28/2025

import logging
import time
import os
import warnings
import torch.nn as nn
import argparse
import torch
import sys
import numpy as np

from typing import Tuple
from models.TGAT import TGAT
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.CAWN import CAWN
from models.TCL import TCL
from models.GraphMixer import GraphMixer
from models.DyGFormer import DyGFormer
from models.modules import MergeLayer, MLPClassifier
from utils.utils import (
    set_random_seed,
    convert_to_gpu,
    get_parameter_sizes,
    get_neighbor_sampler,
    NegativeEdgeSampler,
)
from evaluate_models_utils import (
    evaluate_edge_bank_link_prediction,
)
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data, get_node_classification_data, Data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import load_link_prediction_best_configs

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


def get_explainer_args(is_evaluation: bool) -> argparse.Namespace:
    """
    Get the arguments for the explainer.

    Args:
        None: No arguments are required.

    Returns:
        argparse.Namespace: Arguments for the explainer.
    """
    parser = argparse.ArgumentParser("Interface for the explainer", add_help=True)

    parser.add_argument(
        "--explainer_type",
        type=str,
        default="shapley",
        choices=["shapley", "anchors", "counterfactuals", "LIME", "all"],
        help="name of the explainer",
    )

    # TODO: add functionality for node classification
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="link",
        choices=["link", "node"],
        help="type of prediction task",
    )
    
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="number of samples for explanation background data, higher is more accurate but slower"
    )
    
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=0.01,
        help="ratio of test data to explain, smaller is faster but less accurate"
    )
    
    parser.add_argument(
        "--nsamples",
        type=int,
        default=50,
        help="number of times to re-evaluate model per explanation, higher is more accurate but slower"
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        help="dataset to be used",
        default="wikipedia",
        choices=[
            "wikipedia",
            "reddit",
            "mooc",
            "lastfm",
            "myket",
            "enron",
            "SocialEvo",
            "uci",
            "Flights",
            "CanParl",
            "USLegis",
            "UNtrade",
            "UNvote",
            "Contacts",
        ],
    )
    parser.add_argument("--batch_size", type=int, default=200, help="batch size")
    parser.add_argument(
        "--model_name",
        type=str,
        default="DyGFormer",
        help="name of the model, note that EdgeBank is only applicable for evaluation",
        choices=[
            "JODIE",
            "DyRep",
            "TGAT",
            "TGN",
            "CAWN",
            "EdgeBank",
            "TCL",
            "GraphMixer",
            "DyGFormer",
        ],
    )
    parser.add_argument("--gpu", type=int, default=0, help="number of gpu to use")
    parser.add_argument(
        "--num_neighbors",
        type=int,
        default=20,
        help="number of neighbors to sample for each node",
    )
    parser.add_argument(
        "--sample_neighbor_strategy",
        type=str,
        default="recent",
        choices=["uniform", "recent", "time_interval_aware"],
        help="how to sample historical neighbors",
    )
    parser.add_argument(
        "--time_scaling_factor",
        default=1e-6,
        type=float,
        help="the hyperparameter that controls the sampling preference with time interval, "
        "a large time_scaling_factor tends to sample more on recent links, 0.0 corresponds to uniform sampling, "
        "it works when sample_neighbor_strategy == time_interval_aware",
    )
    parser.add_argument(
        "--num_walk_heads",
        type=int,
        default=8,
        help="number of heads used for the attention in walk encoder",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=2,
        help="number of heads used in attention layer",
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="number of model layers"
    )
    parser.add_argument(
        "--walk_length", type=int, default=1, help="length of each random walk"
    )
    parser.add_argument(
        "--time_gap",
        type=int,
        default=2000,
        help="time gap for neighbors to compute node features",
    )
    parser.add_argument(
        "--time_feat_dim", type=int, default=100, help="dimension of the time embedding"
    )
    parser.add_argument(
        "--position_feat_dim",
        type=int,
        default=172,
        help="dimension of the position embedding",
    )
    parser.add_argument(
        "--edge_bank_memory_mode",
        type=str,
        default="unlimited_memory",
        help="how memory of EdgeBank works",
        choices=["unlimited_memory", "time_window_memory", "repeat_threshold_memory"],
    )
    parser.add_argument(
        "--time_window_mode",
        type=str,
        default="fixed_proportion",
        help="how to select the time window size for time window memory",
        choices=["fixed_proportion", "repeat_interval"],
    )
    parser.add_argument("--patch_size", type=int, default=1, help="patch size")
    parser.add_argument(
        "--channel_embedding_dim",
        type=int,
        default=50,
        help="dimension of each channel embedding",
    )
    parser.add_argument(
        "--max_input_sequence_length",
        type=int,
        default=32,
        help="maximal length of the input sequence of each node",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="learning rate"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        choices=["SGD", "Adam", "RMSprop"],
        help="name of optimizer",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    parser.add_argument(
        "--patience", type=int, default=20, help="patience for early stopping"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.15, help="ratio of validation set"
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.15, help="ratio of test set"
    )
    parser.add_argument("--num_runs", type=int, default=5, help="number of runs")
    parser.add_argument(
        "--test_interval_epochs",
        type=int,
        default=10,
        help="how many epochs to perform testing once",
    )
    parser.add_argument(
        "--negative_sample_strategy",
        type=str,
        default="random",
        choices=["random", "historical", "inductive"],
        help="strategy for the negative edge sampling",
    )
    parser.add_argument(
        "--load_best_configs",
        action="store_true",
        default=False,
        help="whether to load the best configurations",
    )

    try:
        args = parser.parse_args()
        args.device = (
            f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
        )
    except:
        parser.print_help()
        sys.exit()

    if args.model_name == "EdgeBank":
        assert is_evaluation, "EdgeBank is only applicable for evaluation!"

    if args.load_best_configs:
        load_link_prediction_best_configs(args=args)

    return args


def load_data_attributes(
    full_data: Data,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, set, int]:
    """
    Load the data attributes from the full data object. Contains numpy dataset information.

    Args:
        full_data (Data): Data object containing the full dataset information

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, set, int]: Tuple containing the data attributes
    """
    return (
        full_data.src_node_ids,
        full_data.dst_node_ids,
        full_data.node_interact_times,
        full_data.edge_ids,
        full_data.labels,
        full_data.num_interactions,
        full_data.unique_node_ids,
        full_data.num_unique_nodes,
    )


def load_link_prediction_model() -> Tuple[nn.Module, np.ndarray, np.ndarray, Data]:
    """
    load the link prediction model

    Raises:
        ValueError: wrong value for model_name

    Returns:
        Tuple[nn.Module, Data]: Tuple containing the link prediction model and the full dataset information
    """
    # get the args for the link prediction task
    args = get_explainer_args(is_evaluation=True)

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
        seed=1,
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
                exist_ok=True,
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
                    train_data.node_interact_times,
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
                model_name=args.model_name,
            )
            early_stopping.load_checkpoint(model, map_location="cpu")

            model = convert_to_gpu(model, device=args.device)

    return model, node_raw_features, edge_raw_features, full_data


def load_node_classification_model() -> Tuple[nn.Module, np.ndarray, np.ndarray, Data]:
    """
    Load the node classification model.

    Raises:
        ValueError: wrong value for model_name

    Returns:
        Tuple[nn.Module, np.ndarray, np.ndarray, Data]: Tuple containing the node classification model, 
        node_raw_features, edge_raw_features, and the full dataset information
    """
    # get the args for the node classification task
    args = get_explainer_args(is_evaluation=True)

    # get the data for the node classification task
    (
        node_raw_features,
        edge_raw_features,
        full_data,
        train_data,
        val_data,
        test_data,
    ) = get_node_classification_data(
        dataset_name=args.dataset_name,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    # get the neighbor sampler for the node classification task
    full_neighbor_sampler = get_neighbor_sampler(
        data=full_data,
        sample_neighbor_strategy=args.sample_neighbor_strategy,
        time_scaling_factor=args.time_scaling_factor,
        seed=1,
    )

    # load the node classification model (we'll use the first run by default)
    run = 0
    set_random_seed(seed=run)

    args.seed = run
    args.load_model_name = f"node_classification_{args.model_name}_seed{args.seed}"

    # create the folder for the logs
    os.makedirs(
        f"./logs/{args.model_name}/{args.dataset_name}/",
        exist_ok=True,
    )

    # create the file handler for the logs
    fh = logging.FileHandler(
        f"./logs/{args.model_name}/{args.dataset_name}/{str(time.time())}.log"
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
            train_data.node_interact_times,
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

    node_classifier = MLPClassifier(
        input_dim=node_raw_features.shape[1], dropout=args.dropout
    )
    model = nn.Sequential(dynamic_backbone, node_classifier)

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
        model_name=args.model_name,
    )
    early_stopping.load_checkpoint(model, map_location="cpu")

    model = convert_to_gpu(model, device=args.device)
    
    # put the node raw messages of memory-based models on device
    if args.model_name in ["JODIE", "DyRep", "TGN"]:
        for node_id, node_raw_messages in model[
            0
        ].memory_bank.node_raw_messages.items():
            new_node_raw_messages = []
            for node_raw_message in node_raw_messages:
                new_node_raw_messages.append(
                    (node_raw_message[0].to(args.device), node_raw_message[1])
                )
            model[0].memory_bank.node_raw_messages[node_id] = new_node_raw_messages

    return model, node_raw_features, edge_raw_features, full_data
