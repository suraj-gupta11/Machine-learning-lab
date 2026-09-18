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


def find_root_node(feature_matrix, feature_names, target_values, num_bins=4, binning_type="equal_width"):
    info_gain_scores = {}

    for col_index, feature_name in enumerate(feature_names):
        feature_column = [row[col_index] for row in feature_matrix]

        if all(isinstance(val, (int, float)) for val in feature_column):
            feature_column = perform_binning(feature_column, num_bins, binning_type)

        info_gain_scores[feature_name] = calculate_information_gain(feature_column, target_values)

    root_node = max(info_gain_scores, key=info_gain_scores.get)
    return root_node, info_gain_scores


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

    root_node_default, scores_default = find_root_node(feature_matrix, feature_names, target_values)
    print("Root Node (default equal_width, 4 bins):", root_node_default)
    print("Info Gain Scores:", scores_default)

    root_node_custom, scores_custom = find_root_node(feature_matrix, feature_names, target_values, num_bins=3, binning_type="equal_frequency")
    print("\nRoot Node (equal_frequency, 3 bins):", root_node_custom)
    print("Info Gain Scores:", scores_custom)