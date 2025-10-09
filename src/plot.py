import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from math import ceil, sqrt

from config import CLASS_NAMES

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
    plt.xticks(x, [f'{CLASS_NAMES[i]}' for i in train_classes], rotation=20)
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
        if abs(diff) > 0.5:
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
    out = f"{plots_dir}/01_knn_validation_accuracy.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {out}")


def plot_training_curves(hist, title, filename):
    plt.figure(figsize=(7, 4))

    if "loss_history" in hist and len(hist["loss_history"]) > 0:
        loss_history = hist["loss_history"]
        iterations = np.arange(len(loss_history))

        # Plot raw batch loss
        plt.plot(iterations, loss_history, alpha=0.3, color='blue', linewidth=0.5, label="loss (raw)")

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


def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.colorbar()

    tick_labels = [c[:8] for c in CLASS_NAMES]
    plt.xticks(range(len(CLASS_NAMES)), tick_labels, rotation=45, ha='right')
    plt.yticks(range(len(CLASS_NAMES)), tick_labels)

    # Add text annotations
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            plt.text(j, i, str(round(cm[i, j], 2)), ha='center', va='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black')

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {filename}")


def plot_nn_hyperparameter_results(results_list, plots_dir):
    """
    Visualize neural network hyperparameter tuning results.
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
    out = f"{plots_dir}/07_nn_hyperparameter_tuning.png"
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
    out = f"{plots_dir}/12_model_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {out}")


def visualize_grid(Xs, ubound=255.0, padding=1):
    """
    Reshape a 4D tensor of image data to a grid to visualize.

    Inputs:
    - Xs: Data of shape (N, H, W, C)
    - ubound: Output grid will have values scaled to the range [0, ubound]
    - padding: The number of blank pixels between elements of the grid
    """
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


def visualize_nn_weights(net, filename):
    """
    Visualize the learned weights of the first layer of neural network
    """
    W1 = net.params['W1']
    # Reshape weights to image format (28x28x3)
    W1 = W1.reshape(3, 28, 28, -1).transpose(3, 1, 2, 0)

    plt.figure(figsize=(10, 10))
    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
    plt.title('Learned Weights of First Hidden Layer')
    plt.gca().axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {filename}")

def plot_nn_training_loss(history, filename):
    """
    Plot training loss over iterations
    """
    plt.figure(figsize=(8, 4))
    plt.plot(history['loss_history'])
    plt.title('Training loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {filename}")


def plot_nn_accuracy_per_epoch(history, filename):
    """
    Plot training and validation accuracy per epoch
    """
    if len(history['train_acc_history']) > 0:
        plt.figure(figsize=(8, 4))
        plt.plot(history['train_acc_history'], label='train')
        if len(history['val_acc_history']):
            plt.plot(history['val_acc_history'], label='val')
        plt.title('Accuracy per epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {filename}")