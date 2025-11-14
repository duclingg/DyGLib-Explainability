# explain_model.py
# DyGLib-Explainability
# Justin Hoang
# Updated: 11/14/2025

import logging
import os
from explain.utils.load_configs import (
    get_explainer_args,
    load_data_attributes,
    load_link_prediction_model,
)
from explain.shapley import ShapleyExplainer


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger()

    # Get arguments
    args = get_explainer_args(is_evaluation=True)

    logger.info(f"Starting explanation for {args.model_name} on {args.dataset_name}")
    logger.info(f"Explainer type: {args.explainer_type}")
    logger.info(f"Prediction type: {args.prediction_type}")

    if args.prediction_type == "link":
        # Load the link prediction model
        logger.info("Loading link prediction model and data...")
        model, node_raw_features, edge_raw_features, full_data = (
            load_link_prediction_model()
        )

        # Get the data for the explainers
        (
            src_node_ids,
            dst_node_ids,
            node_interact_times,
            edge_ids,
            labels,
            num_interactions,
            unique_node_ids,
            num_unique_nodes,
        ) = load_data_attributes(full_data)

        logger.info(
            f"Data loaded: {num_interactions} interactions, {num_unique_nodes} unique nodes"
        )

        # Create save directories
        save_base_dir = f"./saved_explanations/{args.model_name}/{args.dataset_name}/{args.explainer_type}"
        os.makedirs(save_base_dir, exist_ok=True)

        plots_dir = os.path.join(save_base_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Check explanation type
        if args.explainer_type == "all":
            # TODO: implement multiple explainers
            logger.warning(
                "'all' explainer type not yet implemented. Running Shapley only."
            )
            args.explainer_type = "shapley"

        if args.explainer_type == "shapley":
            logger.info("=" * 80)
            logger.info("Starting SHAP (Shapley) Explanation")
            logger.info("=" * 80)

            shapley_explainer = ShapleyExplainer(
                model=model,
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                num_samples=25,
                sample_ratio=0.01,
                nsamples=25,
                device=args.device,
            )

            # Compute SHAP values
            logger.info("Computing SHAP values...")
            results_path = os.path.join(save_base_dir, "shap_results.json")

            shapley_values = shapley_explainer.compute_shapley_values(
                src_node_ids=src_node_ids,
                dst_node_ids=dst_node_ids,
                node_interact_times=node_interact_times,
                edge_ids=edge_ids,
                labels=labels,
                save_path=results_path,
            )

            # Create visualizations
            logger.info("Creating SHAP visualizations...")
            shapley_explainer.create_visualizations(
                save_dir=plots_dir,
                dataset_name=args.dataset_name,
                model_name=args.model_name,
            )

            # Final summary
            logger.info(f"Results saved to:")
            logger.info(f"JSON:  {results_path}")
            logger.info(f"Plots: {plots_dir}")

        elif args.explainer_type in ["anchors", "counterfactuals", "LIME"]:
            # TODO: implement other explainer types
            logger.error(f"Explainer type '{args.explainer_type}' not yet implemented!")
            raise NotImplementedError(
                f"Explainer type '{args.explainer_type}' is not yet implemented."
            )

    elif args.prediction_type == "node":
        # TODO: implement node classification explanation
        logger.error("Node classification explanation not yet implemented!")
        raise NotImplementedError(
            "Node classification explanation is not yet implemented."
        )

    else:
        logger.error(f"Unknown prediction type: {args.prediction_type}")
        raise ValueError(f"Unknown prediction type: {args.prediction_type}")
