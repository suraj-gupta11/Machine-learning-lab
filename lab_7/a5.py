import math
from collections import Counter

def calculate_entropy(category_labels):
    total_count = len(category_labels)
    label_counts = Counter(category_labels)

    entropy_value = 0.0
    for label, count in label_counts.items():
        probability = count / total_count
        entropy_value -= probability * math.log2(probability)

    return entropy_value


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


def preprocess_feature_matrix(feature_matrix, num_bins=4, binning_type="equal_width"):
    # Converts any continuous (numeric) columns into categorical bin labels.
    # Returns a new matrix where every column is categorical.
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

    # Base case 2: no features left to split on, or max depth reached -> leaf node with majority class
    if len(feature_names) == 0 or (max_depth is not None and current_depth >= max_depth):
        return {"type": "leaf", "class": majority_class(target_values)}

    # Select the best feature to split on using Information Gain
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


if __name__ == "__main__":
    feature_names = ["Outlook", "Temperature", "Humidity", "Windy"]
    feature_matrix = [
        ["Sunny", 85, 85, False],
        ["Sunny", 80, 90, True],
        ["Overcast", 83, 78, False],
        ["Rainy", 70, 96, False],
        ["Rainy", 68, 80, False],
        ["Rainy", 65, 70, True],
        ["Overcast", 64, 65, True],
        ["Sunny", 72, 95, False],
        ["Sunny", 69, 70, False],
        ["Rainy", 75, 80, False],
        ["Sunny", 75, 70, True],
        ["Overcast", 72, 90, True],
        ["Overcast", 81, 75, False],
        ["Rainy", 71, 91, True],
    ]
    target_values = ["No", "No", "Yes", "Yes", "Yes", "No", "Yes",
                      "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"]

    # Step 1: Convert continuous columns to categorical using binning
    processed_matrix = preprocess_feature_matrix(feature_matrix, num_bins=4, binning_type="equal_width")
    print("Processed (binned) Feature Matrix:")
    for row in processed_matrix:
        print(row)

    # Step 2: Build the Decision Tree
    decision_tree = build_decision_tree(processed_matrix, feature_names, target_values)
    print("\nDecision Tree Structure:")
    print(decision_tree)

    # Step 3: Predict on the training samples
    predicted_labels = predict(decision_tree, processed_matrix, feature_names)
    print("\nPredicted Labels:", predicted_labels)
    print("Actual Labels:   ", target_values)

    # Step 4: Check accuracy
    correct_count = sum(1 for p, a in zip(predicted_labels, target_values) if p == a)
    accuracy = correct_count / len(target_values)
    print("\nTraining Accuracy: {:.2f}%".format(accuracy * 100))