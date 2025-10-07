import os
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from data import load_data
from k_nearest_neighbor import KNearestNeighbor
from linear_classifier import LinearClassifier

class_names = [
    'basophil',
    'eosinophil',
    'erythroblast',
    'immature granulocyte',
    'lymphocyte',
    'monocyte',
    'neutrophil',
    'platelet',
]

SEED = 42
np.random.seed(SEED)
random.seed(SEED)


def plot_knn_validation_and_class_distribution(
        plots_dir, k_values, acc_L2, acc_L1, best_params, y_train, y_test
):
    plt.figure(figsize=(18, 6))

    # Left: L1/L2 validation accuracy vs k
    plt.subplot(1, 3, 1)
    plt.plot(k_values, acc_L2, 'bo-', label='L2 distance', linewidth=2, markersize=6)
    plt.plot(k_values, acc_L1, 'rs-', label='L1 distance', linewidth=2, markersize=6)

    best_idx = k_values.index(best_params['k'])
    if best_params['metric'] == 'L2':
        plt.plot(best_params['k'], acc_L2[best_idx], 'go', markersize=12,
                 label=f"Best: k={best_params['k']}, L2")
    else:
        plt.plot(best_params['k'], acc_L1[best_idx], 'go', markersize=12,
                 label=f"Best: k={best_params['k']}, L1")

    plt.xlabel('k (number of neighbors)')
    plt.ylabel('Validation accuracy (%)')
    plt.title('kNN validation accuracy vs k')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(k_values)

    # Middle: L2 - L1 accuracy difference
    plt.subplot(1, 3, 2)
    acc_diff = np.array(acc_L2) - np.array(acc_L1)
    colors = ['blue' if diff > 0 else 'red' for diff in acc_diff]
    bars = plt.bar(range(len(k_values)), acc_diff, alpha=0.7, color=colors)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.xlabel('k Value')
    plt.ylabel('L2 - L1 | Accuracy difference (%)')
    plt.title('Distance comparison')
    plt.xticks(range(len(k_values)), [str(k) for k in k_values], rotation=45)
    plt.grid(True, alpha=0.3, axis='y')

    for bar, diff in zip(bars, acc_diff):
        if abs(diff) > 0.5: # Only label significant differences
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.3 if diff > 0 else -0.5),
                f'{diff:.1f}%',
                ha='center',
                va='bottom' if diff > 0 else 'top',
                fontsize=8
            )

    # Right: class distribution
    plt.subplot(1, 3, 3)
    train_classes, train_counts = np.unique(y_train, return_counts=True)
    test_classes, test_counts = np.unique(y_test, return_counts=True)

    train_percentages = train_counts / len(y_train) * 100
    test_percentages = test_counts / len(y_test) * 100

    x = np.arange(len(train_classes))
    width = 0.35
    plt.bar(x - width / 2, train_percentages, width, label='Training', alpha=0.8)
    plt.bar(x + width / 2, test_percentages, width, label='Test', alpha=0.8)

    plt.xlabel('Class')
    plt.ylabel('Percentage (%)')
    plt.title('Class distribution')
    plt.xticks(x, [f'C{i}' for i in train_classes])
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out = f"{plots_dir}/knn_validation_accuracy_and_class_distribution.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {out}")


def plot_training_curves(hist, title, filename):
    plt.figure(figsize=(7, 4))

    if "loss_history" in hist and len(hist["loss_history"]) > 0:
        loss_history = hist["loss_history"]
        iterations = np.arange(len(loss_history))

        # Plot raw batch loss (noisy)
        plt.plot(iterations, loss_history, alpha=0.3, color='blue', linewidth=0.5, label="Batch loss (raw)")

        # Plot smoothed loss using moving average
        window_size = max(len(loss_history) // 50, 10)
        smoothed_loss = np.convolve(loss_history, np.ones(window_size) / window_size, mode='valid')
        smoothed_iterations = iterations[:len(smoothed_loss)]
        plt.plot(smoothed_iterations, smoothed_loss, color='blue', linewidth=2, label="Loss (smoothed)")

    if "train_acc_history" in hist and len(hist["train_acc_history"]) > 0:
        x_train = np.linspace(0, len(hist["loss_history"]), num=len(hist["train_acc_history"]))
        plt.plot(x_train, hist["train_acc_history"], color='orange', linewidth=2, label="Train acc (epoch)")

    if "val_acc_history" in hist and len(hist["val_acc_history"]) > 0:
        x_val = np.linspace(0, len(hist["loss_history"]), num=len(hist["val_acc_history"]))
        plt.plot(x_val, hist["val_acc_history"], color='green', linewidth=2, label="Val acc (epoch)")

    plt.xlabel("Iteration")
    plt.ylabel("Loss / Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {filename}")


# Use seaborn library instead?
def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.colorbar()

    tick_labels = [c[:8] for c in class_names]
    plt.xticks(range(len(class_names)), tick_labels, rotation=45, ha='right')
    plt.yticks(range(len(class_names)), tick_labels)

    # Add text annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black')

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {filename}")


def main():
    plots_dir = "../plots"
    os.makedirs(plots_dir, exist_ok=True)

    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(size=28, subsample_train=5000)

    print(f"Dataset: {X_train.shape[0]} train, {X_val.shape[0]} val, {X_test.shape[0]} test")
    print(f"Features: {X_train.shape[1]}, Classes: {len(np.unique(y_train))}")
    print()

    # =========================
    # PART 1: kNN
    # =========================
    knn = KNearestNeighbor()
    knn.train(X_train, y_train)

    k_values = [1, 3, 5, 7, 13, 21, 33, 45, 55, 67, 79, 91, 111, 131, 149, 167, 193, 201]
    distance_metrics = ['L2', 'L1']

    print("Hyperparameter tuning results:")
    print("k\tL2 accuracy\tL1 accuracy")
    print("-" * 35)

    acc_L2 = []
    acc_L1 = []
    best_accuracy = 0.0
    best_params = {}

    for k in k_values:
        results = {}
        for metric in distance_metrics:
            preds = knn.predict(X_val, k=k, metric=metric)
            acc = np.mean(preds == y_val) * 100
            results[metric] = acc
            if acc > best_accuracy:
                best_accuracy = acc
                best_params = {'k': k, 'metric': metric}
        acc_L2.append(results['L2'])
        acc_L1.append(results['L1'])
        print(f"{k}\t{results['L2']:.1f}%\t\t{results['L1']:.1f}%")

    plot_knn_validation_and_class_distribution(
        plots_dir, k_values, acc_L2, acc_L1, best_params, y_train, y_test
    )

    print(f"\nBest: k={best_params['k']}, {best_params['metric']} ({best_accuracy:.1f}%)")
    final_predictions = knn.predict(X_test, k=best_params['k'], metric=best_params['metric'])
    test_accuracy = np.mean(final_predictions == y_test) * 100
    print(f"Final test accuracy: {test_accuracy:.1f}%")

    print("\nClass distribution:")
    for dataset_name, labels in [("Training", y_train), ("Test", y_test)]:
        print(f"{dataset_name}:", end=" ")
        classes, counts = np.unique(labels, return_counts=True)
        percentages = [f"{count / len(labels) * 100:.1f}%" for count in counts]
        print(", ".join(f"C{cls}: {pct}" for cls, pct in zip(classes, percentages)))

    print("\nAdditional Statistics:")
    print(f"Training set size: {len(y_train)}")
    print(f"Validation set size: {len(y_val)}")
    print(f"Test set size: {len(y_test)}")
    print(f"Feature dimensions: {X_train.shape[1]}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    print(f"Best k value: {best_params['k']}")
    print(f"Best distance metric: {best_params['metric']}")

    # ===========================================
    # PART 2: Linear Classifiers (SVM and Softmax)
    # ===========================================
    print("\n" + "=" * 60)
    print("PART 2: Linear Classifiers")
    print("=" * 60)

    num_train, dim = X_train.shape
    num_classes = int(np.max(y_train) + 1)

    def run_grid(loss_type, Xtr, ytr, Xva, yva,
                 lrs, regs, num_iters=200, batch_size=200, print_every=100):
        best_val = -1.0
        best_tuple = None
        best_hist = None

        print(f"\nOptimizing {loss_type} classifier:")
        print("lr\treg\tVal Accuracy")
        print("-" * 35)

        for lr in lrs:
            for reg in regs:
                clf = LinearClassifier(input_dim=dim, num_classes=num_classes, loss_type=loss_type)
                hist = clf.train(
                    Xtr, ytr,
                    X_val=Xva, y_val=yva,
                    learning_rate=lr, reg=reg,
                    num_iters=num_iters, batch_size=batch_size,
                    print_every=print_every
                )

                if hist["val_acc_history"]:
                    val_acc = hist["val_acc_history"][-1]
                else:
                    val_acc = np.mean(clf.predict(Xva) == yva)

                print(f"{lr:<7g}\t{reg:<7g}\t{val_acc * 100:.2f}%")

                if val_acc > best_val:
                    best_val = val_acc
                    best_tuple = (clf, lr, reg)
                    best_hist = hist

        clf, lr, reg = best_tuple
        print(f"\nBest {loss_type.capitalize()}: lr={lr}, reg={reg} ({best_val * 100:.2f}%)\n")
        return clf, lr, reg, best_hist

    lrs = [1e-4, 5e-4, 1e-3]
    regs = [1e-4, 5e-4, 1e-3, 5e-3]

    # SVM
    svm_model, best_lr_svm, best_reg_svm, svm_hist = run_grid(
        "svm", X_train, y_train, X_val, y_val,
        lrs, regs, num_iters=200, batch_size=200, print_every=100
    )
    y_test_pred_svm = svm_model.predict(X_test)
    svm_test_acc = np.mean(y_test_pred_svm == y_test) * 100
    print(f"Final SVM test accuracy: {svm_test_acc:.2f}%")

    # Softmax
    softmax_model, best_lr_soft, best_reg_soft, softmax_hist = run_grid(
        "softmax", X_train, y_train, X_val, y_val,
        lrs, regs, num_iters=200, batch_size=200, print_every=100
    )
    y_test_pred_soft = softmax_model.predict(X_test)
    softmax_test_acc = np.mean(y_test_pred_soft == y_test) * 100
    print(f"Final Softmax test accuracy: {softmax_test_acc:.2f}%")

    # Training curves
    plot_training_curves(svm_hist, "SVM Training (loss + epoch acc)", f"{plots_dir}/svm_training_curves.png")
    plot_training_curves(softmax_hist, "Softmax Training (loss + epoch acc)",
                         f"{plots_dir}/softmax_training_curves.png")

    # Confusion matrices
    plot_confusion_matrix(y_test, y_test_pred_svm, "SVM Confusion Matrix", f"{plots_dir}/confusion_matrix_svm.png")
    plot_confusion_matrix(y_test, y_test_pred_soft, "Softmax Confusion Matrix",
                          f"{plots_dir}/confusion_matrix_softmax.png")

    print("\nDone!")


if __name__ == "__main__":
    main()