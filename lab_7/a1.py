import math
from collections import Counter

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


if __name__ == "__main__":
    continuous_outcome_data = [12.5, 15.3, 22.1, 30.8, 45.6,
                                18.2, 25.7, 40.1, 33.3, 10.9,
                                27.4, 38.5, 19.9, 42.2, 21.0]

    binned_data = equal_width_binning(continuous_outcome_data, num_bins=4)
    print("Binned Data:", binned_data)

    dataset_entropy = calculate_entropy(binned_data)
    print("Entropy of the dataset: {:.4f} bits".format(dataset_entropy))

    categorical_outcome_data = ["Yes", "No", "Yes", "Yes", "No", "No", "Yes"]
    categorical_entropy = calculate_entropy(categorical_outcome_data)
    print("Entropy of categorical dataset: {:.4f} bits".format(categorical_entropy))