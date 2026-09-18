import math
from collections import Counter

# ---------- A1: Binning & Entropy ----------

def equal_width_binning(data_values, num_bins=4):
    min_value = min(data_values)
    max_value = max(data_values)
    bin_width = (max_value - min_value) / num_bins

    bin_labels = []
    for value in data_values:
        if value == max_value:
            bin_index = num_bins - 1
        else:
            bin_index = int((value - min_value) / bin_width)
        bin_labels.append(f"Bin_{bin_index + 1}")

    return bin_labels


def calculate_entropy(category_labels):
    total_count = len(category_labels)
    label_counts = Counter(category_labels)

    entropy_value = 0.0
    for label, count in label_counts.items():
        probability = count / total_count
        entropy_value -= probability * math.log2(probability)

    return entropy_value


# ---------- A2: Gini Index ----------

def calculate_gini_index(category_labels):
    total_count = len(category_labels)
    label_counts = Counter(category_labels)

    sum_of_squares = 0.0
    for label, count in label_counts.items():
        probability = count / total_count
        sum_of_squares += probability ** 2

    gini_value = 1 - sum_of_squares
    return gini_value


# ---------- A3/A4: Information Gain & Flexible Binning ----------

def perform_binning(data_values, num_bins=4, binning_type="equal_width"):
    if binning_type == "equal_width":
        min_value = min(data_values)
        max_value = max(data_values)
        bin_width = (max_value - min_value) / num_bins

        bin_labels = []
        for value in data_values:
            if value == max_value:
                bin_index = num_bins - 1
            else:
                bin_index = int((value - min_value) / bin_width)
            bin_labels.append(f"Bin_{bin_index + 1}")

    elif binning_type == "equal_frequency":
        sorted_values = sorted(data_values)
        total_count = len(sorted_values)
        points_per_bin = total_count / num_bins

        bin_boundaries = [sorted_values[int(i * points_per_bin)] for i in range(1, num_bins)]

        bin_labels = []
        for value in data_values:
            bin_index = 0
            for boundary in bin_boundaries:
                if value > boundary:
                    bin_index += 1
                else:
                    break
            bin_labels.append(f"Bin_{bin_index + 1}")

    else:
        raise ValueError("binning_type must be 'equal_width' or 'equal_frequency'")

    return bin_labels


def calculate_information_gain(feature_values, target_values):
    total_count = len(target_values)
    total_entropy = calculate_entropy(target_values)

    unique_features = set(feature_values)
    weighted_entropy = 0.0

    for feature_val in unique_features:
        subset_targets = [target_values[i] for i in range(total_count) if feature_values[i] == feature_val]
        weight = len(subset_targets) / total_count
        weighted_entropy += weight * calculate_entropy(subset_targets)

    information_gain = total_entropy - weighted_entropy
    return information_gain


def find_root_node(feature_matrix, feature_names, target_values, num_bins=4, binning_type="equal_width"):
    info_gain_scores = {}

    for col_index, feature_name in enumerate(feature_names):
        feature_column = [row[col_index] for row in feature_matrix]

        if all(isinstance(val, (int, float)) for val in feature_column):
            feature_column = perform_binning(feature_column, num_bins, binning_type)

        info_gain_scores[feature_name] = calculate_information_gain(feature_column, target_values)

    root_node = max(info_gain_scores, key=info_gain_scores.get)
    return root_node, info_gain_scores


# ---------- A5: Full Decision Tree ----------

def preprocess_feature_matrix(feature_matrix, num_bins=4, binning_type="equal_width"):
    # Converts any continuous (numeric) columns into categorical bin labels.
    num_columns = len(feature_matrix[0])
    processed_matrix = [list(row) for row in feature_matrix]

    for col_index in range(num_columns):
        column_values = [row[col_index] for row in feature_matrix]
        if all(isinstance(val, (int, float)) for val in column_values):
            binned_column = perform_binning(column_values, num_bins, binning_type)
            for row_index in range(len(processed_matrix)):
                processed_matrix[row_index][col_index] = binned_column[row_index]

    return processed_matrix


def majority_class(target_values):
    class_counts = Counter(target_values)
    return class_counts.most_common(1)[0][0]


def select_best_feature(feature_matrix, feature_names, target_values):
    info_gain_scores = {}

    for col_index, feature_name in enumerate(feature_names):
        feature_column = [row[col_index] for row in feature_matrix]
        info_gain_scores[feature_name] = calculate_information_gain(feature_column, target_values)

    best_feature_name = max(info_gain_scores, key=info_gain_scores.get)
    best_feature_index = feature_names.index(best_feature_name)
    return best_feature_name, best_feature_index, info_gain_scores


def build_decision_tree(feature_matrix, feature_names, target_values, max_depth=None, current_depth=0):
    # Base case 1: all target values belong to a single class -> leaf node
    if len(set(target_values)) == 1:
        return {"type": "leaf", "class": target_values[0]}

    # Base case 2: no features left, or max depth reached -> leaf node with majority class
    if len(feature_names) == 0 or (max_depth is not None and current_depth >= max_depth):
        return {"type": "leaf", "class": majority_class(target_values)}

    best_feature_name, best_feature_index, info_gain_scores = select_best_feature(
        feature_matrix, feature_names, target_values
    )

    tree_node = {
        "type": "node",
        "feature": best_feature_name,
        "default_class": majority_class(target_values),
        "children": {}
    }

    unique_values = set(row[best_feature_index] for row in feature_matrix)
    remaining_feature_names = [name for i, name in enumerate(feature_names) if i != best_feature_index]

    for value in unique_values:
        subset_indices = [i for i in range(len(feature_matrix)) if feature_matrix[i][best_feature_index] == value]
        subset_matrix = [
            [row[i] for i in range(len(row)) if i != best_feature_index]
            for row in (feature_matrix[i] for i in subset_indices)
        ]
        subset_targets = [target_values[i] for i in subset_indices]

        tree_node["children"][value] = build_decision_tree(
            subset_matrix, remaining_feature_names, subset_targets, max_depth, current_depth + 1
        )

    return tree_node


def predict_sample(tree, sample, feature_names):
    if tree["type"] == "leaf":
        return tree["class"]

    feature_index = feature_names.index(tree["feature"])
    feature_value = sample[feature_index]

    if feature_value in tree["children"]:
        remaining_feature_names = [name for name in feature_names if name != tree["feature"]]
        remaining_sample = [sample[i] for i in range(len(sample)) if i != feature_index]
        return predict_sample(tree["children"][feature_value], remaining_sample, remaining_feature_names)
    else:
        return tree["default_class"]


def predict(tree, sample_matrix, feature_names):
    predictions = []
    for sample in sample_matrix:
        predictions.append(predict_sample(tree, sample, feature_names))
    return predictions


# ---------- A6: Visualization Helpers ----------

def add_nodes_edges(graph, tree, parent_id=None, edge_label="", node_counter=[0]):
    node_id = str(node_counter[0])
    node_counter[0] += 1

    if tree["type"] == "leaf":
        graph.node(node_id, label=f"Class: {tree['class']}", shape="box", style="filled", fillcolor="lightgreen")
    else:
        graph.node(node_id, label=tree["feature"], shape="ellipse", style="filled", fillcolor="lightblue")

    if parent_id is not None:
        graph.edge(parent_id, node_id, label=edge_label)

    if tree["type"] == "node":
        for value, child_tree in tree["children"].items():
            add_nodes_edges(graph, child_tree, node_id, str(value), node_counter)

    return graph


def visualize_decision_tree(tree, output_filename="decision_tree"):
    from graphviz import Digraph
    graph = Digraph(comment="Decision Tree")
    add_nodes_edges(graph, tree)
    graph.render(output_filename, format="png", cleanup=True)
    return output_filename + ".png"


# ---------- A7: Two-Feature Decision Boundary Helpers ----------

def compute_bin_boundaries(data_values, num_bins=4):
    min_value = min(data_values)
    max_value = max(data_values)
    bin_width = (max_value - min_value) / num_bins
    return min_value, max_value, bin_width


def assign_bin_label(value, min_value, max_value, bin_width, num_bins):
    if value >= max_value:
        bin_index = num_bins - 1
    elif value <= min_value:
        bin_index = 0
    else:
        bin_index = int((value - min_value) / bin_width)
    return f"Bin_{bin_index + 1}"


def build_two_feature_tree(feature1_values, feature2_values, target_values, num_bins=4):
    bounds1 = compute_bin_boundaries(feature1_values, num_bins)
    bounds2 = compute_bin_boundaries(feature2_values, num_bins)

    binned1 = [assign_bin_label(v, *bounds1, num_bins) for v in feature1_values]
    binned2 = [assign_bin_label(v, *bounds2, num_bins) for v in feature2_values]

    feature_matrix = [[binned1[i], binned2[i]] for i in range(len(target_values))]
    feature_names = ["Feature1", "Feature2"]

    tree = build_decision_tree(feature_matrix, feature_names, target_values)
    return tree, bounds1, bounds2


def predict_two_feature_point(tree, feature1_value, feature2_value, bounds1, bounds2, num_bins, feature_names):
    label1 = assign_bin_label(feature1_value, *bounds1, num_bins)
    label2 = assign_bin_label(feature2_value, *bounds2, num_bins)
    return predict_sample(tree, [label1, label2], feature_names)