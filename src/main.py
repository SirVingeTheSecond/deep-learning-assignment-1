from data import load_data
from k_nearest_neighbor import KNearestNeighbor
import numpy as np


def main():
    # Load and display dataset info
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(size=28, subsample_train=5000)

    print(f"Dataset: {X_train.shape[0]} train, {X_val.shape[0]} val, {X_test.shape[0]} test")
    print(f"Features: {X_train.shape[1]}, Classes: {len(np.unique(y_train))}")
    print()

    # Init and train KNN
    knn = KNearestNeighbor()
    knn.train(X_train, y_train)

    # Hyperparameter
    k_values = [1, 3, 5, 7, 13, 21, 33, 45, 55, 67, 79, 91, 111, 131, 149, 167, 193, 201]
    distance_metrics = ['L2', 'L1']

    print("Hyperparameter tuning results:")
    print("k\tL2 Accuracy\tL1 Accuracy")
    print("-" * 35)

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

        print(f"{k}\t{results['L2']:.1f}%\t\t{results['L1']:.1f}%")

    # Final evaluation
    print(f"\nBest: k={best_params['k']}, {best_params['metric']} ({best_accuracy:.1f}%)")

    final_predictions = knn.predict(X_test, k=best_params['k'], distance_metric=best_params['metric'])
    test_accuracy = np.mean(final_predictions == y_test) * 100
    print(f"Final test accuracy: {test_accuracy:.1f}%")

    # Show class distribution
    print("\nClass distribution:")
    for dataset_name, labels in [("Training", y_train), ("Test", y_test)]:
        print(f"{dataset_name}:", end=" ")
        classes, counts = np.unique(labels, return_counts=True)
        percentages = [f"{count / len(labels) * 100:.1f}%" for count in counts]
        print(", ".join(f"C{cls}: {pct}" for cls, pct in zip(classes, percentages)))


if __name__ == "__main__":
    main()