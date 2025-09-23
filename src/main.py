from data import load_data
from k_nearest_neighbor import KNearestNeighbor
import numpy as np
import matplotlib.pyplot as plt
import os


def main():
    plots_dir = "../plots/"
    os.makedirs(plots_dir, exist_ok=True)

    # Load and display dataset info
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(size=28, subsample_train=5000)

    print(f"Dataset: {X_train.shape[0]} train, {X_val.shape[0]} val, {X_test.shape[0]} test")
    print(f"Features: {X_train.shape[1]}, Classes: {len(np.unique(y_train))}")
    print()

    # Init and train kNN
    knn = KNearestNeighbor()
    knn.train(X_train, y_train)

    # Hyperparameter
    k_values = [1, 3, 5, 7, 13, 21, 33, 45, 55, 67, 79, 91, 111, 131, 149, 167, 193, 201]
    distance_metrics = ['L2', 'L1']

    print("Hyperparameter tuning results:")
    print("k\tL2 accuracy\tL1 accuracy")
    print("-" * 35)

    # Store results for plotting
    acc_L2 = []
    acc_L1 = []

    best_accuracy = 0
    best_params = {}

    # Test all combinations
    for k in k_values:
        results = {}

        for metric in distance_metrics:
            predictions = knn.predict(X_val, k=k, distance_metric=metric)
            accuracy = np.mean(predictions == y_val) * 100
            results[metric] = accuracy

            # Track best configuration
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {'k': k, 'metric': metric}

        # Store accuracies for plotting
        acc_L2.append(results['L2'])
        acc_L1.append(results['L1'])

        print(f"{k}\t{results['L2']:.1f}%\t\t{results['L1']:.1f}%")

    # Validation accuracy vs k and class distribution
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(k_values, acc_L2, 'bo-', label='L2 distance', linewidth=2, markersize=6)
    plt.plot(k_values, acc_L1, 'rs-', label='L1 distance', linewidth=2, markersize=6)

    # Best config
    if best_params['metric'] == 'L2':
        best_idx = k_values.index(best_params['k'])
        plt.plot(best_params['k'], acc_L2[best_idx], 'go', markersize=12,
                 label=f"Best: k={best_params['k']}, L2")
    else:
        best_idx = k_values.index(best_params['k'])
        plt.plot(best_params['k'], acc_L1[best_idx], 'go', markersize=12,
                 label=f"Best: k={best_params['k']}, L1")

    plt.xlabel('k (number of neighbors)', fontsize=12)
    plt.ylabel('Validation accuracy (%)', fontsize=12)
    plt.title('kNN validation accuracy vs k', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(k_values)

    # Class Distribution
    plt.subplot(1, 2, 2)

    # Get class distributions
    train_classes, train_counts = np.unique(y_train, return_counts=True)
    test_classes, test_counts = np.unique(y_test, return_counts=True)

    train_percentages = train_counts / len(y_train) * 100
    test_percentages = test_counts / len(y_test) * 100

    x = np.arange(len(train_classes))
    width = 0.35

    plt.bar(x - width / 2, train_percentages, width, label='Training', alpha=0.8)
    plt.bar(x + width / 2, test_percentages, width, label='Test', alpha=0.8)

    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.title('Class distribution', fontsize=14)
    plt.xticks(x, [f'C{i}' for i in train_classes])
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f"{plots_dir}/knn_validation_accuracy_and_class_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {plots_dir}/knn_validation_accuracy_and_class_distribution.png")

    # Last eval
    print(f"\nBest: k={best_params['k']}, {best_params['metric']} ({best_accuracy:.1f}%)")

    final_predictions = knn.predict(X_test, k=best_params['k'], distance_metric=best_params['metric'])
    test_accuracy = np.mean(final_predictions == y_test) * 100
    print(f"Final test accuracy: {test_accuracy:.1f}%")

    # Best performance analysis
    plt.figure(figsize=(12, 6))

    # Plot best performance for each k
    best_acc_per_k = [max(l2, l1) for l2, l1 in zip(acc_L2, acc_L1)]

    plt.subplot(1, 2, 1)
    plt.plot(k_values, best_acc_per_k, 'go-', linewidth=2, markersize=6)
    plt.axhline(y=best_accuracy, color='r', linestyle='--', alpha=0.7,
                label=f'Best overall: {best_accuracy:.1f}%')
    plt.xlabel('k (number of neighbors)', fontsize=12)
    plt.ylabel('Best validation accuracy (%)', fontsize=12)
    plt.title('Best accuracy per k value', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(k_values)

    # Performance difference between L1 and L2
    plt.subplot(1, 2, 2)
    acc_diff = np.array(acc_L2) - np.array(acc_L1)
    colors = ['blue' if diff > 0 else 'red' for diff in acc_diff]
    bars = plt.bar(range(len(k_values)), acc_diff, alpha=0.7, color=colors)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.xlabel('k Value', fontsize=12)
    plt.ylabel('L1 vs L2 | Accuracy difference (%)', fontsize=12)
    plt.title('Distance comparison', fontsize=14)
    plt.xticks(range(len(k_values)), [str(k) for k in k_values])
    plt.grid(True, alpha=0.3, axis='y')

    # value labels on bars
    for i, (bar, diff) in enumerate(zip(bars, acc_diff)):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (0.1 if diff > 0 else -0.3),
                 f'{diff:.1f}%', ha='center', va='bottom' if diff > 0 else 'top', fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{plots_dir}/knn_performance_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {plots_dir}/knn_performance_analysis.png")

    # Class distribution
    print("\nClass distribution:")
    for dataset_name, labels in [("Training", y_train), ("Test", y_test)]:
        print(f"{dataset_name}:", end=" ")
        classes, counts = np.unique(labels, return_counts=True)
        percentages = [f"{count / len(labels) * 100:.1f}%" for count in counts]
        print(", ".join(f"C{cls}: {pct}" for cls, pct in zip(classes, percentages)))

    # Additional statistics
    print(f"\nAdditional Statistics:")
    print(f"Training set size: {len(y_train)}")
    print(f"Validation set size: {len(y_val)}")
    print(f"Test set size: {len(y_test)}")
    print(f"Feature dimensions: {X_train.shape[1]}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    print(f"Best k value: {best_params['k']}")
    print(f"Best distance metric: {best_params['metric']}")

if __name__ == "__main__":
    main()