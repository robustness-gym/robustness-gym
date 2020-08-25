import torch
import numpy as np
import graph_tool as gt
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from graph_tool.draw import graph_draw
from joblib import Parallel, delayed

from typing import List, Dict, Tuple, Optional
from influence_utils import faiss_utils
from influence_utils import parallel
from experiments.visualization_utils import (
    get_circle_coordinates,
    get_within_circle_constraint,
    # distance_to_points_on_circle,
    # distance_to_points_within_circle,
    distance_to_points_within_circle_vectorized)
from experiments import constants
from experiments import misc_utils
from experiments.hans_utils import HansHelper
from transformers import Trainer, TrainingArguments


KNN_K = 1000


def main(
    train_task_name: str,
    eval_task_name: str,
    num_eval_to_collect: int,
    hans_heuristic: Optional[str] = None,
) -> List[Dict[int, float]]:

    if train_task_name not in ["mnli-2", "hans"]:
        raise ValueError

    if eval_task_name not in ["mnli-2", "hans"]:
        raise ValueError

    tokenizer, model = misc_utils.create_tokenizer_and_model(
        constants.MNLI2_MODEL_PATH)

    train_dataset, _ = misc_utils.create_datasets(
        task_name=train_task_name,
        tokenizer=tokenizer)

    _, eval_dataset = misc_utils.create_datasets(
        task_name=eval_task_name,
        tokenizer=tokenizer)

    if train_task_name == "mnli-2":
        faiss_index = faiss_utils.FAISSIndex(768, "Flat")
        faiss_index.load(constants.MNLI2_FAISS_INDEX_PATH)
    else:
        faiss_index = None

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./tmp-output",
            per_device_train_batch_size=128,
            per_device_eval_batch_size=128,
            learning_rate=5e-5,
            logging_steps=100),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    if eval_task_name in ["mnli-2"]:
        eval_instance_data_loader = misc_utils.get_dataloader(
            dataset=eval_dataset,
            batch_size=1,
            random=False)

    if eval_task_name in ["hans"]:
        if hans_heuristic is None:
            raise ValueError("`hans_heuristic` cannot be None for now")

        hans_helper = HansHelper(
            hans_train_dataset=None,
            hans_eval_dataset=eval_dataset)

        _, eval_instance_data_loader = hans_helper.get_dataset_and_dataloader_of_heuristic(
            mode="eval",
            heuristic=hans_heuristic,
            batch_size=1,
            random=False)

    # Data-points where the model got wrong
    wrong_input_collections = []
    for i, test_inputs in enumerate(eval_instance_data_loader):
        logits, labels, step_eval_loss = misc_utils.predict(
            trainer=trainer,
            model=model,
            inputs=test_inputs)
        if logits.argmax(axis=-1).item() != labels.item():
            wrong_input_collections.append(test_inputs)

    params_filter = [
        n for n, p in model.named_parameters()
        if not p.requires_grad]

    weight_decay_ignores = [
        "bias",
        "LayerNorm.weight"] + [
        n for n, p in model.named_parameters()
        if not p.requires_grad]

    if eval_task_name == "mnli-2":
        s_test_damp = 5e-3
        s_test_scale = 1e4
        s_test_num_samples = 1000

    if eval_task_name == "hans":
        s_test_damp = 5e-3
        s_test_scale = 1e6
        s_test_num_samples = 1000

    influences_collections = []
    for index, inputs in enumerate(wrong_input_collections[:num_eval_to_collect]):
        print(f"#{index}")
        if faiss_index is not None:
            features = misc_utils.compute_BERT_CLS_feature(model, **inputs)
            features = features.cpu().detach().numpy()
            KNN_distances, KNN_indices = faiss_index.search(
                k=KNN_K, queries=features)
        else:
            KNN_indices = None

        # influences, _, _ = nn_influence_utils.compute_influences(
        #     n_gpu=1,
        #     device=torch.device("cuda"),
        #     batch_train_data_loader=batch_train_data_loader,
        #     instance_train_data_loader=instance_train_data_loader,
        #     model=model,
        #     test_inputs=inputs,
        #     params_filter=params_filter,
        #     weight_decay=constants.WEIGHT_DECAY,
        #     weight_decay_ignores=weight_decay_ignores,
        #     s_test_damp=s_test_damp,
        #     s_test_scale=s_test_scale,
        #     s_test_num_samples=s_test_num_samples,
        #     train_indices_to_include=KNN_indices,
        #     precomputed_s_test=None)

        influences, _ = parallel.compute_influences_parallel(
            # Avoid clash with main process
            device_ids=[0, 1, 2, 3],
            train_dataset=train_dataset,
            batch_size=1,
            model=model,
            test_inputs=inputs,
            params_filter=params_filter,
            weight_decay=constants.WEIGHT_DECAY,
            weight_decay_ignores=weight_decay_ignores,
            s_test_damp=s_test_damp,
            s_test_scale=s_test_scale,
            s_test_num_samples=s_test_num_samples,
            train_indices_to_include=KNN_indices,
            return_s_test=False,
            debug=False)

        influences_collections.append(influences)

    return influences_collections


def run_experiments(option: str) -> List[List[Dict[int, float]]]:
    if option == "mnli2_and_hans":
        mnli2_influences = main(
            train_task_name="mnli-2",
            eval_task_name="mnli-2",
            num_eval_to_collect=100)

        hans_influences = main(
            train_task_name="mnli-2",
            eval_task_name="hans",
            num_eval_to_collect=100)

        return [mnli2_influences, hans_influences]

    if option == "mnli_and_hans_heuristic":
        hans_influences_collections = []
        for hans_heuristic in ["lexical_overlap", "subsequence", "constituent"]:
            hans_influences = main(
                train_task_name="mnli-2",
                eval_task_name="hans",
                num_eval_to_collect=100,
                hans_heuristic=hans_heuristic)

            hans_influences_collections.append(hans_influences)

        return hans_influences_collections

    if option == "hans_and_hans_heuristic":
        hans_influences_collections = []
        for hans_heuristic in ["lexical_overlap", "subsequence", "constituent"]:
            hans_influences = main(
                train_task_name="hans",
                eval_task_name="hans",
                num_eval_to_collect=100,
                hans_heuristic=hans_heuristic)

            hans_influences_collections.append(hans_influences)

        return hans_influences_collections

    raise ValueError(f"Unrecognized `option` {option}")


def get_datapoints_map(
    influences_collections: List[Dict[int, float]]
) -> Tuple[List[int], Dict[int, int]]:
    possible_datapoints = []
    for influences in influences_collections:
        possible_datapoints.extend(list(influences.keys()))

    possible_datapoints = sorted(set(possible_datapoints))
    datapoints_map = dict((v, k) for k, v in enumerate(possible_datapoints))
    return possible_datapoints, datapoints_map


def get_graph(
        influences_collections_list: List[List[Dict[int, float]]]
) -> gt.Graph:

    influences_collections_list_flatten = []
    for influences_collections in influences_collections_list:
        # Assume they all have the same lengths
        if len(influences_collections_list[0][0]) != len(influences_collections[0]):
            raise ValueError
        influences_collections_list_flatten.extend(influences_collections)

    # Note they share the same training dataset
    possible_datapoints, datapoints_map = get_datapoints_map(
        influences_collections=influences_collections_list_flatten)

    g = gt.Graph(directed=True)
    # Edge properties
    e_colors = g.new_edge_property("int")
    e_weights = g.new_edge_property("double")
    e_signed_influences = g.new_edge_property("double")
    e_unsigned_influences = g.new_edge_property("double")
    # Vertex properties
    v_sizes = g.new_vertex_property("int")
    v_colors = g.new_vertex_property("int")
    v_data_indices = g.new_vertex_property("string")
    v_positions = g.new_vertex_property("vector<double>")
    v_positive_positions = g.new_vertex_property("vector<double>")
    v_negative_positions = g.new_vertex_property("vector<double>")

    train_vertices = []
    eval_vertices_collections = []

    NUM_INFLUENCE_COLLECTIONS = len(influences_collections_list)
    VERTEX_COLORS = range(2, 2 + NUM_INFLUENCE_COLLECTIONS)
    HELPFUL_EDGE_COLOR = 0
    HARMFUL_EDGE_COLOR = 1
    TRAIN_VERTEX_SIZE = 3
    TRAIN_VERTEX_COLOR = 0
    EVAL_VERTICES_RADIUS = 3
    TRAIN_VERTICES_RADIUS = 2

    # Add train vertices
    for datapoint_index in trange(len(possible_datapoints)):
        v = g.add_vertex()
        v_sizes[v] = TRAIN_VERTEX_SIZE
        v_colors[v] = TRAIN_VERTEX_COLOR
        v_data_indices[v] = f"train-{datapoint_index}"
        train_vertices.append(v)

    # Add eval vertices
    for i, influences_collections in enumerate(influences_collections_list):

        eval_vertices = []
        for datapoint_index in trange(len(influences_collections)):
            v = g.add_vertex()
            v_sizes[v] = 10
            v_colors[v] = VERTEX_COLORS[i]
            v_data_indices[v] = f"A_eval-{datapoint_index}"

            base_degree = (360 / NUM_INFLUENCE_COLLECTIONS) * i
            fine_degree = (360 / NUM_INFLUENCE_COLLECTIONS / len(influences_collections)) * datapoint_index
            x_y_coordinate = get_circle_coordinates(
                r=EVAL_VERTICES_RADIUS,
                degree=base_degree + fine_degree)
            position = np.random.normal(x_y_coordinate, 0.1)
            v_positions[v] = position
            v_positive_positions[v] = position
            v_negative_positions[v] = position
            eval_vertices.append(v)

        eval_vertices_collections.append(eval_vertices)

    # Add edges
    def add_edges(influences_collections: List[Dict[int, float]],
                  eval_vertices: List[gt.Vertex]) -> None:
        for eval_index, influences in enumerate(tqdm(influences_collections)):
            for train_index, train_influence in influences.items():
                # Negative influence is helpful (when the prediction is wrong)
                if train_influence < 0.0:
                    train_vertex = train_vertices[datapoints_map[train_index]]
                    eval_vertex = eval_vertices[eval_index]
                    e = g.add_edge(train_vertex, eval_vertex)
                    e_colors[e] = HELPFUL_EDGE_COLOR
                    e_weights[e] = np.abs(train_influence)
                    e_signed_influences[e] = train_influence
                    e_unsigned_influences[e] = np.abs(train_influence)
                else:
                    train_vertex = train_vertices[datapoints_map[train_index]]
                    eval_vertex = eval_vertices[eval_index]
                    e = g.add_edge(train_vertex, eval_vertex)
                    e_colors[e] = HARMFUL_EDGE_COLOR
                    e_weights[e] = np.abs(train_influence)
                    e_signed_influences[e] = train_influence
                    e_unsigned_influences[e] = np.abs(train_influence)

    for i, influences_collections in enumerate(influences_collections_list):
        add_edges(influences_collections, eval_vertices_collections[i])

    def _calculate_position(train_vertex: gt.Vertex) -> None:
        """Determine X-axis and Y-axis
        - We use X-axis to determine the divergence
        - We use Y-axis to determine the helpfulness/harmfulness
        """
        # Two types of targets
        # two types of connections
        _positive_points = []
        _negative_points = []
        _positive_influences = []
        _negative_influences = []
        for e in train_vertex.all_edges():
            target = e.target()
            if e_signed_influences[e] > 0:
                _positive_points.append(v_positions[target])
                _positive_influences.append(e_unsigned_influences[e])
            else:
                _negative_points.append(v_positions[target])
                _negative_influences.append(e_unsigned_influences[e])

        # `minimize` might fail using `np.sqrt(2)` for some reasons :\
        bound = 1.4 * TRAIN_VERTICES_RADIUS
        constraints = ({
            "type": "ineq",
            "fun": get_within_circle_constraint(TRAIN_VERTICES_RADIUS)
        })

        if len(_positive_influences) == 0:
            _positive_xval = 0.0
            _positive_yval = 0.0
        else:
            _positive_points_stacked = np.stack(_positive_points, axis=0)
            _positive_influences_stacked = np.stack(_positive_influences, axis=0)
            _positive_optimize_result = minimize(
                distance_to_points_within_circle_vectorized,
                x0=(0, 0),
                constraints=constraints,
                bounds=((-bound, bound), (-bound, bound)),
                args=(_positive_influences_stacked,
                      _positive_points_stacked))
            _positive_xval, _positive_yval = _positive_optimize_result.x

        if len(_negative_influences) == 0:
            _negative_xval = 0.0
            _negative_yval = 0.0
        else:
            _negative_points_stacked = np.stack(_negative_points, axis=0)
            _negative_influences_stacked = np.stack(_negative_influences, axis=0)
            _negative_optimize_result = minimize(
                distance_to_points_within_circle_vectorized,
                x0=(0, 0),
                constraints=constraints,
                bounds=((-bound, bound), (-bound, bound)),
                args=(_negative_influences_stacked,
                      _negative_points_stacked))
            _negative_xval, _negative_yval = _negative_optimize_result.x

        _positive_xval = np.random.normal(_positive_xval, 0.01)
        _negative_xval = np.random.normal(_negative_xval, 0.01)
        _positive_yval = np.random.normal(_positive_yval, 0.01)
        _negative_yval = np.random.normal(_negative_yval, 0.01)
        v_positive_positions[train_vertex] = np.array([_positive_xval, _positive_yval])
        v_negative_positions[train_vertex] = np.array([_negative_xval, _negative_yval])
        v_positions[train_vertex] = np.array([(_positive_xval + _negative_xval) / 2,
                                              (_positive_yval + _negative_yval) / 2])

    # Run them in parallel
    # Parallel(n_jobs=-1)(
    #     delayed(_calculate_position)(train_vertex)
    #     for train_vertex in tqdm(train_vertices))
    for train_vertex in tqdm(train_vertices):
        _calculate_position(train_vertex)

    # Assign Edge properties
    g.edge_properties["colors"] = e_colors
    g.edge_properties["weights"] = e_weights
    g.edge_properties["signed_influences"] = e_signed_influences
    g.edge_properties["unsigned_influences"] = e_unsigned_influences
    # Assign Vertex properties
    g.vertex_properties["sizes"] = v_sizes
    g.vertex_properties["colors"] = v_colors
    g.vertex_properties["data_indices"] = v_data_indices
    g.vertex_properties["positions"] = v_positions
    g.vertex_properties["positive_positions"] = v_positive_positions
    g.vertex_properties["negative_positions"] = v_negative_positions

    return g, {
        "train_vertices": train_vertices,
        "eval_vertices_collections": eval_vertices_collections
    }


def get_recall_plot(model, example, faiss_index, full_influences_dict):
    # plt.rcParams["figure.figsize"] = [20, 5]
    recall_num_neighbors = [10, 100, 1000]
    num_neighbors = [10, 100, 1000, 10000, 50000, 100000]
    names = ["Most Helpful",
             "Most Harmful",
             "Most Influencetial",
             "Least Influential"]

    features = misc_utils.compute_BERT_CLS_feature(model, **example)
    features = features.cpu().detach().numpy()
    if list(full_influences_dict.keys()) != list(range(len(full_influences_dict))):
        raise ValueError

    full_influences = []
    for key in sorted(full_influences_dict):
        full_influences.append(full_influences_dict[key])

    sorted_indices_small_to_large = np.argsort(full_influences)
    sorted_indices_large_to_small = np.argsort(full_influences)[::-1]
    sorted_indices_abs_large_to_small = np.argsort(np.abs(full_influences))[::-1]
    sorted_indices_abs_small_to_large = np.argsort(np.abs(full_influences))

    fig, axes = plt.subplots(1, 4, sharex=True, sharey=True)
    recalls_collections = {}
    for i, (name, sorted_indices) in enumerate(zip(
            names,
            [sorted_indices_small_to_large,
             sorted_indices_large_to_small,
             sorted_indices_abs_large_to_small,
             sorted_indices_abs_small_to_large])):

        recalls_collection = []
        for recall_k in tqdm(recall_num_neighbors):
            recalls = []
            influential = sorted_indices[:recall_k]
            influential_set = set(influential.tolist())
            for k in num_neighbors:
                distances, indices = faiss_index.search(k=k, queries=features)
                indices_set = set(indices.squeeze(axis=0).tolist())
                recall = len(influential_set & indices_set) / len(influential_set)
                recalls.append(recall)

            recalls_collection.append(recalls)
            axes[i].plot(num_neighbors, recalls,
                         linestyle="--", marker="o",
                         label=f"recall@{recall_k}")

        axes[i].legend()
        axes[i].set_title(name)
        axes[i].set_xscale("log")
        axes[i].set_ylabel("Recall")
        axes[i].set_xlabel("Number of Nearest Neighbors")
        recalls_collections[name] = recalls_collection

    return recalls_collections
