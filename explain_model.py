# explain_model.py
# DyGLib-Explainability
# Justin Hoang
# 10/28/2025

import logging
from explain.utils.load_configs import (
    get_explainer_args,
    load_data_attributes,
    load_link_prediction_model,
)
from explain.shapley import ShapleyExplainer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    args = get_explainer_args(is_evaluation=True)

    if args.prediction_type == "link":
        # load the link prediction model
        model, node_raw_features, edge_raw_features, full_data = (
            load_link_prediction_model()
        )

        # get the data for the explainers
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

        # check explanation type
        if args.explainer_type == "all":
            # TODO: implement
            pass
        elif args.explainer_type == "shapley":
            shapley_explainer = ShapleyExplainer(model)

            shapley_values = shapley_explainer.compute_shapley_values(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                src_node_ids=src_node_ids,
                dst_node_ids=dst_node_ids,
                node_interact_times=node_interact_times,
                labels=labels,
                edge_ids=edge_ids,
            )

            logger.info(f"Shapley values: {shapley_values}")
