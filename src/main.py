import os
import numpy as np
import random
import matplotlib.pyplot as plt
from imageio.v2 import sizes
from sklearn.metrics import confusion_matrix

from data import load_data
from k_nearest_neighbor import KNearestNeighbor
from linear_classifier import LinearClassifier
from neural_network import FullyConnectedNN, load_data_nn

class_names = [
    'Basop',
    'Eosin',
    'Eryth',
    'Immat',
    'Lymph',
    'Monoc',
    'Neutr',
    'Plate',
]

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

def plot_class_distribution(plots_dir, y_train, y_test, y_val):
    plt.plot()
    train_classes, train_counts = np.unique(y_train, return_counts=True)
    test_classes, test_counts = np.unique(y_test, return_counts=True)
    val_classes, val_counts = np.unique(y_val, return_counts=True)

    train_percentages = train_counts / len(y_train) * 100
    test_percentages = test_counts / len(y_test) * 100
    val_percentages = val_counts / len(y_val) * 100

    x = np.arange(len(train_classes))
    width = 0.2
    plt.bar(x - width * 1.1, train_percentages, width, label='Training', alpha=0.8)
    plt.bar(x, test_percentages, width, label='Test', alpha=0.8)
    plt.bar(x + width * 1.1, val_percentages, width, label='Validation', alpha=0.8)
    

    plt.xlabel('Class')
    plt.ylabel('Percentage (%)')
    plt.title('Class distribution')
    plt.xticks(x, [f'{class_names[i]}' for i in train_classes], rotation=20)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out = f"{plots_dir}/dataset_class_distribution.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()

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
    cm = confusion_matrix(y_true, y_pred, normalize='true')
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
            plt.text(j, i, str(round(cm[i, j], 2)), ha='center', va='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black')

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {filename}")


def visualize_nn_weights(net, filename):
    """
    Visualize the learned weights of the first layer of neural network
    """
    W1 = net.params['W1']
    # Reshape weights to image format (28x28x3)
    W1 = W1.reshape(3, 28, 28, -1).transpose(3, 1, 2, 0)

    plt.figure(figsize=(10, 10))
    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
    plt.title('Learned weights of first hidden layer')
    plt.gca().axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {filename}")


def plot_nn_hyperparameter_results(results_list, plots_dir):
    """
    Visualize neural network hyperparameter tuning results.
    Similar style to kNN visualization with multiple subplots.
    """
    fig = plt.figure(figsize=(18, 12))

    # Extract unique values
    hidden_sizes = sorted(set(r['hidden'] for r in results_list))
    learning_rates = sorted(set(r['lr'] for r in results_list))
    reg_strengths = sorted(set(r['reg'] for r in results_list))
    optimizers = sorted(set(r['opt'] for r in results_list))

    # Plot 1: Hidden size vs validation accuracy (grouped by optimizer)
    ax1 = plt.subplot(2, 3, 1)
    for opt in optimizers:
        opt_results = [r for r in results_list if r['opt'] == opt]
        hidden_acc = {}
        for r in opt_results:
            if r['hidden'] not in hidden_acc:
                hidden_acc[r['hidden']] = []
            hidden_acc[r['hidden']].append(r['val_acc'])

        sizes = sorted(hidden_acc.keys())
        avg_accs = [np.mean(hidden_acc[s]) * 100 for s in sizes]
        ax1.plot(sizes, avg_accs, 'o-', label=opt, linewidth=2, markersize=8)

    ax1.set_xlabel('Hidden Layer Size')
    ax1.set_ylabel('Validation Accuracy (%)')
    ax1.set_title('Hidden Size vs Validation Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Learning rate vs validation accuracy
    ax2 = plt.subplot(2, 3, 2)
    for opt in optimizers:
        opt_results = [r for r in results_list if r['opt'] == opt]
        lr_acc = {}
        for r in opt_results:
            if r['lr'] not in lr_acc:
                lr_acc[r['lr']] = []
            lr_acc[r['lr']].append(r['val_acc'])

        lrs = sorted(lr_acc.keys())
        avg_accs = [np.mean(lr_acc[lr]) * 100 for lr in lrs]
        ax2.semilogx(lrs, avg_accs, 'o-', label=opt, linewidth=2, markersize=8)

    ax2.set_xlabel('Learning Rate (log scale)')
    ax2.set_ylabel('Validation Accuracy (%)')
    ax2.set_title('Learning Rate vs Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Regularization strength vs validation accuracy
    ax3 = plt.subplot(2, 3, 3)
    for opt in optimizers:
        opt_results = [r for r in results_list if r['opt'] == opt]
        reg_acc = {}
        for r in opt_results:
            if r['reg'] not in reg_acc:
                reg_acc[r['reg']] = []
            reg_acc[r['reg']].append(r['val_acc'])

        regs = sorted(reg_acc.keys())
        avg_accs = [np.mean(reg_acc[reg]) * 100 for reg in regs]
        ax3.semilogx(regs, avg_accs, 'o-', label=opt, linewidth=2, markersize=8)

    ax3.set_xlabel('Regularization Strength (log scale)')
    ax3.set_ylabel('Validation Accuracy (%)')
    ax3.set_title('Regularization vs Validation Accuracy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Optimizer comparison (box plot)
    ax4 = plt.subplot(2, 3, 4)
    opt_data = []
    for opt in optimizers:
        opt_accs = [r['val_acc'] * 100 for r in results_list if r['opt'] == opt]
        opt_data.append(opt_accs)

    bp = ax4.boxplot(opt_data, tick_labels=optimizers, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax4.set_ylabel('Validation Accuracy (%)')
    ax4.set_title('Optimizer Comparison (Distribution)')
    ax4.grid(True, alpha=0.3, axis='y')

    # Plot 5: Top 15 configurations
    ax5 = plt.subplot(2, 3, 5)
    sorted_results = sorted(results_list, key=lambda x: x['val_acc'], reverse=True)[:15]
    config_labels = [f"H{r['hidden']}_L{r['lr']:.0e}_R{r['reg']:.0e}_{r['opt'][:3]}"
                     for r in sorted_results]
    val_accs = [r['val_acc'] * 100 for r in sorted_results]

    colors = ['green' if i == 0 else 'blue' for i in range(len(val_accs))]
    bars = ax5.barh(range(len(val_accs)), val_accs, color=colors, alpha=0.7)
    ax5.set_yticks(range(len(val_accs)))
    ax5.set_yticklabels(config_labels, fontsize=7)
    ax5.set_xlabel('Validation Accuracy (%)')
    ax5.set_title('Top 15 Configurations')
    ax5.grid(True, alpha=0.3, axis='x')
    ax5.invert_yaxis()

    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, val_accs)):
        ax5.text(acc + 0.2, bar.get_y() + bar.get_height() / 2,
                 f'{acc:.2f}%', va='center', fontsize=7)

    # Plot 6: Accuracy distribution histogram
    ax6 = plt.subplot(2, 3, 6)
    all_accs = [r['val_acc'] * 100 for r in results_list]
    ax6.hist(all_accs, bins=15, color='steelblue', alpha=0.7, edgecolor='black')
    ax6.axvline(np.mean(all_accs), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_accs):.2f}%')
    ax6.axvline(np.max(all_accs), color='green', linestyle='--', linewidth=2, label=f'Best: {np.max(all_accs):.2f}%')
    ax6.set_xlabel('Validation Accuracy (%)')
    ax6.set_ylabel('Number of Configurations')
    ax6.set_title('Accuracy Distribution Across All Configurations')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out = f"{plots_dir}/nn_hyperparameter_tuning.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {out}")


def plot_model_comparison(knn_acc, svm_acc, softmax_acc, nn_acc, plots_dir):
    """
    Create a bar chart comparing all models
    """
    models = ['kNN', 'SVM', 'Softmax', 'Neural Network']
    accuracies = [knn_acc, svm_acc, softmax_acc, nn_acc]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    bars = ax.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Model Comparison on Test Set', fontsize=14, fontweight='bold')
    ax.set_ylim([0, max(accuracies) + 5])
    ax.grid(True, alpha=0.3, axis='y')

    # Highlight best model
    best_idx = accuracies.index(max(accuracies))
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)

    plt.tight_layout()
    out = f"{plots_dir}/model_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {out}")


def visualize_grid(Xs, ubound=255.0, padding=1):
    """
    Reshape a 4D tensor of image data to a grid for easy visualization.

    Inputs:
    - Xs: Data of shape (N, H, W, C)
    - ubound: Output grid will have values scaled to the range [0, ubound]
    - padding: The number of blank pixels between elements of the grid
    """
    from math import ceil, sqrt

    (N, H, W, C) = Xs.shape
    grid_size = int(ceil(sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C))
    next_idx = 0
    y0, y1 = 0, H
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if next_idx < N:
                img = Xs[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    return grid

def main():
    plots_dir = "../plots"
    os.makedirs(plots_dir, exist_ok=True)

    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(size=28, subsample_train=5000)

    print(f"Dataset: {X_train.shape[0]} train, {X_val.shape[0]} val, {X_test.shape[0]} test")
    print(f"Features: {X_train.shape[1]}, Classes: {len(np.unique(y_train))}")
    print()

    plot_class_distribution(plots_dir, y_train, y_test, y_val)

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


    plot_confusion_matrix(y_test, final_predictions, "KNN Confusion Matrix", f"{plots_dir}/confusion_matrix_knn.png")

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

    # ===========================================
    # PART 3: Neural Networks
    # ===========================================
    print("\n" + "=" * 60)
    print("PART 3: Neural Networks")
    print("=" * 60)

    print("\nLoading data for Neural Network...")
    X_train_nn, y_train_nn, X_val_nn, y_val_nn, X_test_nn, y_test_nn = load_data_nn(
        size=28, subsample_train=5000, subsample_val=500, seed=SEED
    )
    print(f"NN data shapes: {X_train_nn.shape}, {X_val_nn.shape}, {X_test_nn.shape}")

    def run_nn_grid(Xtr, ytr, Xva, yva, hidden_sizes, lrs, regs, optimizers,
                    num_iters=200, batch_size=64, print_every=100):
        """
        Hyperparameter search for neural network.
        Similar to run_grid for linear classifiers but with more hyperparameters.
        """
        best_val = -1.0
        best_tuple = None
        best_hist = None
        all_results = []

        print("\nOptimizing Neural Network:")
        print("Hidden\tLR\tReg\tOptimizer\tVal Accuracy")
        print("-" * 60)

        # Use input dimension from NN data (no bias column)
        input_dim = Xtr.shape[1]

        for hidden_size in hidden_sizes:
            for lr in lrs:
                for reg in regs:
                    for opt in optimizers:
                        layers = [input_dim, hidden_size, num_classes]
                        nn = FullyConnectedNN(
                            layers=layers,
                            reg_strength=reg,
                            loss='softmax',
                            seed=SEED
                        )

                        hist = nn.train(
                            Xtr, ytr,
                            X_val=Xva, y_val=yva,
                            learning_rate=lr, reg=reg,
                            num_iters=num_iters, batch_size=batch_size,
                            optimizer=opt, print_every=print_every
                        )

                        if hist["val_acc_history"]:
                            val_acc = hist["val_acc_history"][-1]
                        else:
                            val_acc = np.mean(nn.predict(Xva) == yva)

                        print(f"{hidden_size}\t{lr:<7g}\t{reg:<7g}\t{opt}\t\t{val_acc * 100:.2f}%")

                        # Track all results for visualization
                        all_results.append({
                            'hidden': hidden_size,
                            'lr': lr,
                            'reg': reg,
                            'opt': opt,
                            'val_acc': val_acc
                        })

                        if val_acc > best_val:
                            best_val = val_acc
                            best_tuple = (nn, hidden_size, lr, reg, opt)
                            best_hist = hist

        nn, h_size, lr, reg, opt = best_tuple
        print(f"\nBest NN: hidden={h_size}, lr={lr}, reg={reg}, opt={opt} ({best_val * 100:.2f}%)\n")
        return nn, h_size, lr, reg, opt, best_hist, all_results

    # Grid search parameters
    hidden_sizes = [100, 200, 500]
    nn_lrs = [0.001, 0.005, 0.01]
    nn_regs = [0.0001, 0.001, 0.01]
    optimizers = ['sgd', 'momentum', 'adam']

    nn_model, best_hidden, best_lr_nn, best_reg_nn, best_opt, nn_hist, nn_results = run_nn_grid(
        X_train_nn, y_train_nn, X_val_nn, y_val_nn,
        hidden_sizes, nn_lrs, nn_regs, optimizers,
        num_iters=200, batch_size=64, print_every=100
    )

    y_test_pred_nn = nn_model.predict(X_test_nn)
    nn_test_acc = np.mean(y_test_pred_nn == y_test_nn) * 100
    print(f"Final Neural Network test accuracy: {nn_test_acc:.2f}%")

    # Visualize hyperparameter tuning results
    plot_nn_hyperparameter_results(nn_results, plots_dir)

    # Training curves for neural network
    plot_training_curves(nn_hist, "Neural Network Training (loss + epoch acc)",
                         f"{plots_dir}/nn_training_curves.png")

    # Confusion matrix for neural network
    plot_confusion_matrix(y_test_nn, y_test_pred_nn, "Neural Network Confusion Matrix",
                          f"{plots_dir}/confusion_matrix_nn.png")

    # Visualize learned weights
    visualize_nn_weights(nn_model, f"{plots_dir}/nn_learned_weights.png")

    # Per class accuracy analysis
    print("\nPer-class accuracy on test set:")
    for i, class_name in enumerate(class_names):
        class_mask = y_test_nn == i
        if np.sum(class_mask) > 0:
            class_acc = np.mean(y_test_pred_nn[class_mask] == y_test_nn[class_mask]) * 100
            print(f"  {class_name:25s}: {class_acc:5.2f}% ({np.sum(class_mask):4d} samples)")

    # statistics for neural network
    total_params = sum(p.size for p in nn_model.params.values())
    print(f"\nNeural Network statistics:")
    print(f"  Architecture: {nn_model.layers}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Best hidden layer size: {best_hidden}")
    print(f"  Best learning rate: {best_lr_nn}")
    print(f"  Best regularization: {best_reg_nn}")
    print(f"  Best optimizer: {best_opt}")
    if nn_hist['val_acc_history']:
        best_epoch = np.argmax(nn_hist['val_acc_history'])
        print(f"  Best validation accuracy at epoch: {best_epoch}")
        print(f"  Final train accuracy: {nn_hist['train_acc_history'][-1] * 100:.2f}%")
        print(f"  Final validation accuracy: {nn_hist['val_acc_history'][-1] * 100:.2f}%")
        print(f"  Train-Val gap: {(nn_hist['train_acc_history'][-1] - nn_hist['val_acc_history'][-1]) * 100:.2f}%")

    # ===========================================
    # COMPARISON
    # ===========================================
    print("\n" + "=" * 60)
    print("FINAL COMPARISON OF ALL MODELS")
    print("=" * 60)
    print(f"kNN:            {test_accuracy:.2f}%")
    print(f"SVM:            {svm_test_acc:.2f}%")
    print(f"Softmax:        {softmax_test_acc:.2f}%")
    print(f"Neural Network: {nn_test_acc:.2f}%")
    print("=" * 60)

    plot_model_comparison(test_accuracy, svm_test_acc, softmax_test_acc, nn_test_acc, plots_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()