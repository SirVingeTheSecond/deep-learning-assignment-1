from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from math import ceil, sqrt
from scipy.interpolate import griddata
import os

from config import CLASS_NAMES, LINEAR_LEARNING_RATES, LINEAR_REGULARIZATIONS


# HELPER FUNCTIONS

def _ensure_plot_dir(filepath):
    """Internal helper to ensure directory exists before saving plot"""
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _save_and_close(filepath, dpi=300):
    """
    Consistently save and close plots with proper error handling.

    Args:
        filepath: Full path to save the plot
        dpi: Resolution for saved image (default: 300)
    """
    _ensure_plot_dir(filepath)
    plt.tight_layout()
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {filepath}")


# ============================================================================
# GENERAL
# ============================================================================

def plot_class_distribution(plots_dir, y_train, y_test, y_val):
    """Plot class distribution across train, test, and validation sets"""
    plt.figure()
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

    _save_and_close(f"{plots_dir}/dataset_class_distribution.png")


def plot_confusion_matrix(y_true, y_pred, title, filepath):
    """Create and save confusion matrix visualization"""
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

    _save_and_close(filepath)


# ============================================================================
# K-NEAREST NEIGHBORS
# ============================================================================

def plot_knn_validation_and_class_distribution(
        plots_dir, k_values, acc_L2, acc_L1, best_params, y_train, y_test
):
    """Plot kNN validation accuracy for different k values and distance metrics"""
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

    _save_and_close(f"{plots_dir}/01_knn_validation_accuracy.png")


# ============================================================================
# LINEAR CLASSIFIER
# ============================================================================

def plot_linear_classifier_learned_weights(plots_dir, model_type, weights):
    """Visualize learned weights for linear classifiers in 2x4 grid"""
    # Remove bias and transpose if needed
    W = weights.transpose()[:, :-1]  # shape: (num_classes, input_dim)

    fig, ax = plt.subplots(2, 4, figsize=(12, 6))

    for i in range(W.shape[0]):
        img = W[i].reshape(28, 28, 3)

        row, col = divmod(i, 4)  # Determine position in 2x4 grid
        ax[row, col].imshow(img * 6)  # Scale color up a bit
        ax[row, col].set_title(CLASS_NAMES[i], fontsize=10)
        ax[row, col].axis('off')

    fig.suptitle(f"Learned Weights for {model_type} Classifier", fontsize=14)

    _save_and_close(f"{plots_dir}/linear_{model_type}_weights_visualized.png")


def plot_linear_classifier_hyperparameters(plots_dir, results, loss_type):
    """Create hyperparameter surface plot for linear classifiers"""
    # Extract columns
    x = np.array([r[0] for r in results])  # learning rates
    y = np.array([r[1] for r in results])  # regularizations
    z = np.array([r[2] for r in results])  # accuracies

    # Interpolate in log-space
    log_x = np.log10(x)
    log_y = np.log10(y)
    xi = np.linspace(log_x.min(), log_x.max(), 200)
    yi = np.linspace(log_y.min(), log_y.max(), 200)
    Xi, Yi = np.meshgrid(xi, yi)

    Zi = griddata((log_x, log_y), z, (Xi, Yi), method='linear')

    fig, ax = plt.subplots()

    # Show gradient background
    im = ax.imshow(
        Zi,
        extent=(log_x.min(), log_x.max(), log_y.min(), log_y.max()),
        origin='lower',
        cmap='plasma',
        aspect='auto',
        vmin=1 / 8,
        vmax=1
    )

    # Overlay original scatter points
    sc = ax.scatter(log_x, log_y, c=z, cmap='plasma', edgecolor='black', s=100, vmin=1 / 8, vmax=1)

    best_idx = np.argmax(z)
    ax.scatter(log_x[best_idx], log_y[best_idx], s=200, edgecolor='yellow', facecolor='none', linewidth=2)

    # Colorbar
    plt.colorbar(im, ax=ax, label='Validation Accuracy')

    # Log-scale ticks
    x_ticks = np.array(LINEAR_LEARNING_RATES)
    ax.set_xticks(np.log10(x_ticks))
    ax.set_xticklabels(["{:.0e}".format(v) for v in x_ticks])

    y_ticks = np.array(LINEAR_REGULARIZATIONS)
    ax.set_yticks(np.log10(y_ticks))
    ax.set_yticklabels(["{:.0e}".format(v) for v in y_ticks])

    plt.xlabel("Learning Rate")
    plt.ylabel("Regularization Strength")
    plt.title(f"Hyperparameter Surface ({loss_type})")

    _save_and_close(f"{plots_dir}/08_linear_hyperparam_results_{loss_type}.png")


def plot_training_curves(hist, title, filepath):
    """Plot training loss and accuracy curves"""
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

    _save_and_close(filepath)


# ============================================================================
# MODEL COMPARISON
# ============================================================================

def plot_model_comparison(knn_acc, svm_acc, softmax_acc, nn_acc, plots_dir):
    """Create a bar chart comparing all models"""
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

    _save_and_close(f"{plots_dir}/12_model_comparison.png")


# ============================================================================
# NEURAL NETWORK
# ============================================================================

def visualize_grid(Xs, ubound=255.0, padding=1):
    """
    Reshape a 4D tensor of image data to a grid for visualization.

    Args:
        Xs: Data of shape (N, H, W, C)
        ubound: Output grid will have values scaled to the range [0, ubound]
        padding: The number of blank pixels between elements of the grid

    Returns:
        grid: Grid representation of images
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


def visualize_nn_weights(net, plots_dir):
    """
    ENHANCED: Visualize the learned weights with improved handling of narrow bottlenecks.
    Creates a combined visualization showing both first and second layer weights.
    """
    fig = plt.figure(figsize=(18, 14))

    # ============================================================
    # PLOT 1: First Layer Weights (Feature Detectors)
    # ============================================================
    ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=4)
    W1 = net.params['W1']
    W1_reshaped = W1.reshape(3, 28, 28, -1).transpose(3, 1, 2, 0)

    # Show a grid of the first layer features
    grid_img = visualize_grid(W1_reshaped, padding=3)
    ax1.imshow(grid_img.astype('uint8'))
    ax1.set_title('First Layer Weights (Feature Detectors)',
                  fontsize=14, fontweight='bold', pad=20)
    ax1.axis('off')

    # Add text explaining what we're seeing
    n_features = W1_reshaped.shape[0]
    hidden_size = net.layers[1]
    ax1.text(0.5, -0.05,
             f'Showing {n_features} learned feature detectors from input layer (hidden size: {hidden_size})',
             transform=ax1.transAxes, ha='center', fontsize=10, style='italic')

    # Add title for second section
    fig.text(0.5, 0.42, 'Output Layer Weights (Class-Specific Patterns)',
             ha='center', fontsize=14, fontweight='bold')

    # IMPROVED: Add warning if bottleneck is too narrow
    if hidden_size < 100:
        fig.text(0.5, 0.39,
                 f'⚠ Note: Narrow hidden layer ({hidden_size} neurons) limits spatial reconstruction quality',
                 ha='center', fontsize=10, style='italic', color='red')
    else:
        fig.text(0.5, 0.39,
                 'Reconstructed input space patterns emphasized by each class',
                 ha='center', fontsize=10, style='italic')

    # ============================================================
    # PLOT 2: Second Layer Weights - IMPROVED METHOD
    # ============================================================
    W2 = net.params['W2']  # Shape: (hidden_size, num_classes)

    for i in range(min(8, W2.shape[1])):
        row = 2 + (i // 4)  # Start from row 2, two rows for 8 classes
        col = i % 4
        ax = plt.subplot2grid((4, 4), (row, col))

        # Get weights for this class from hidden layer
        class_weights = W2[:, i]

        # IMPROVED: Use top-k neurons weighted combination for better reconstruction
        if hidden_size < 100:
            # For narrow layers, use top contributing neurons
            k = min(20, hidden_size)
            top_indices = np.argsort(np.abs(class_weights))[-k:]

            reconstructed = np.zeros(W1.shape[0])
            for idx in top_indices:
                reconstructed += class_weights[idx] * W1[:, idx]
        else:
            # For wider layers, use all neurons
            reconstructed = np.dot(W1, class_weights)

        img = reconstructed.reshape(28, 28, 3)

        # Normalize for visualization
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)

        ax.imshow(img_norm)
        ax.set_title(CLASS_NAMES[i], fontsize=10, fontweight='bold')
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save_and_close(f"{plots_dir}/11_nn_learned_weights_comprehensive.png")


def visualize_nn_weights_comparison(net, plots_dir):
    """
    ENHANCED: Create a direct comparison visualization matching the linear classifier format.
    Shows only the output layer weights in the same 2x4 grid, with improved reconstruction.
    """
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    W1 = net.params['W1']  # (input_dim, hidden_size)
    W2 = net.params['W2']  # (hidden_size, num_classes)
    hidden_size = net.layers[1]

    for i in range(8):
        row, col = divmod(i, 4)
        ax = axes[row, col]

        # Get weights for this class
        class_weights = W2[:, i]

        # IMPROVED: Handle narrow bottlenecks better
        if hidden_size < 100:
            # Use weighted top-k approach
            k = min(20, hidden_size)
            top_indices = np.argsort(np.abs(class_weights))[-k:]

            reconstructed = np.zeros(W1.shape[0])
            for idx in top_indices:
                reconstructed += class_weights[idx] * W1[:, idx]
        else:
            # Standard reconstruction for wider networks
            reconstructed = np.dot(W1, class_weights)

        img = reconstructed.reshape(28, 28, 3)

        # Normalize and scale similar to linear classifier visualization
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)

        ax.imshow(img_norm * 255)  # Scale to match linear classifier
        ax.set_title(CLASS_NAMES[i], fontsize=10)
        ax.axis('off')

    # IMPROVED: Add informative title
    title = "Neural Network Output Layer Weights (Class-Specific)"
    if hidden_size < 100:
        title += f"\n(Narrow bottleneck: {hidden_size} neurons limits detail)"
    fig.suptitle(title, fontsize=14, fontweight='bold')

    _save_and_close(f"{plots_dir}/11b_nn_weights_linear_comparison.png")


def visualize_nn_weights_simple(net, filepath):
    """
    Simple visualization of first layer weights only.
    Maintains backward compatibility with existing code.
    """
    W1 = net.params['W1']
    W1 = W1.reshape(3, 28, 28, -1).transpose(3, 1, 2, 0)

    plt.figure(figsize=(10, 10))
    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
    plt.title('Learned Weights of First Hidden Layer', fontsize=14, fontweight='bold')
    plt.gca().axis('off')

    _save_and_close(filepath)


def plot_nn_training_loss(history, filepath):
    """Plot training loss over iterations"""
    plt.figure(figsize=(8, 4))
    plt.plot(history['loss_history'])
    plt.title('Training loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)

    _save_and_close(filepath)


def plot_nn_accuracy_per_epoch(history, filepath, optimizer=None, params=None):
    """Plot training and validation accuracy per epoch"""
    if len(history['train_acc_history']) > 0:
        plt.figure(figsize=(8, 4))
        plt.plot(history['train_acc_history'], label='train')
        if len(history['val_acc_history']):
            plt.plot(history['val_acc_history'], label='val')

        title = 'Accuracy per epoch'
        if optimizer:
            title += f' (Optimizer: {optimizer}'
            if params:
                title += f', LR: {params.get("lr", "?")}, Reg: {params.get("reg", "?")}'
            title += ')'

        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        _save_and_close(filepath)


def plot_nn_hyperparameter_results(results_list, plots_dir):
    """Visualize neural network hyperparameter tuning results in a 2x3 grid"""
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

    _save_and_close(f"{plots_dir}/07_nn_hyperparameter_tuning.png")


def plot_nn_hyperparameter_surfaces(results_list, plots_dir):
    """
    ENHANCED: Create hyperparameter surface plots for neural networks with better data visibility.
    Shows LR vs Reg interaction for each optimizer with larger markers and value labels.
    """
    optimizers = sorted(set(r['opt'] for r in results_list))

    # Create a figure with subplots for each optimizer
    n_opts = len(optimizers)
    fig, axes = plt.subplots(1, n_opts, figsize=(6 * n_opts, 5))

    if n_opts == 1:
        axes = [axes]

    for idx, opt in enumerate(optimizers):
        ax = axes[idx]

        # Filter results for this optimizer
        opt_results = [r for r in results_list if r['opt'] == opt]

        if not opt_results:
            continue

        # Extract data
        x = np.array([r['lr'] for r in opt_results])
        y = np.array([r['reg'] for r in opt_results])
        z = np.array([r['val_acc'] * 100 for r in opt_results])

        # Count unique values for grid info
        unique_lrs = sorted(set(r['lr'] for r in opt_results))
        unique_regs = sorted(set(r['reg'] for r in opt_results))
        n_points = len(opt_results)

        # Create grid for interpolation
        log_x = np.log10(x)
        log_y = np.log10(y)
        xi = np.linspace(log_x.min(), log_x.max(), 200)
        yi = np.linspace(log_y.min(), log_y.max(), 200)
        Xi, Yi = np.meshgrid(xi, yi)

        # Interpolate
        Zi = griddata((log_x, log_y), z, (Xi, Yi), method='linear')

        # Plot gradient background
        im = ax.imshow(
            Zi,
            extent=(log_x.min(), log_x.max(), log_y.min(), log_y.max()),
            origin='lower',
            cmap='plasma',
            aspect='auto',
            vmin=max(z.min() - 5, 0),
            vmax=min(z.max() + 5, 100)
        )

        # ENHANCED: Larger scatter points with white edges
        sc = ax.scatter(log_x, log_y, c=z, cmap='plasma',
                        edgecolor='white', s=300, linewidth=2,
                        vmin=max(z.min() - 5, 0),
                        vmax=min(z.max() + 5, 100),
                        zorder=10)

        # ENHANCED: Add value labels on each point
        for i, (lx, ly, val) in enumerate(zip(log_x, log_y, z)):
            ax.text(lx, ly, f'{val:.1f}',
                    ha='center', va='center',
                    fontsize=8, fontweight='bold',
                    color='white' if val < 70 else 'black',
                    zorder=11)

        # Highlight best point
        best_idx = np.argmax(z)
        ax.scatter(log_x[best_idx], log_y[best_idx],
                   s=400, edgecolor='yellow', facecolor='none',
                   linewidth=3, label=f'Best: {z[best_idx]:.2f}%',
                   zorder=12)

        ax.set_xticks(np.log10(unique_lrs))
        ax.set_xticklabels([f'{lr:.0e}' for lr in unique_lrs], rotation=45)
        ax.set_yticks(np.log10(unique_regs))
        ax.set_yticklabels([f'{reg:.0e}' for reg in unique_regs])

        ax.set_xlabel('Learning Rate (log scale)')
        ax.set_ylabel('Regularization Strength (log scale)')
        ax.set_title(f'Optimizer: {opt}')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # ENHANCED: Add grid size info
        ax.text(0.02, 0.98, f'Grid: {len(unique_lrs)}×{len(unique_regs)} = {n_points} points',
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=9, zorder=15)

        # Add colorbar
        plt.colorbar(im, ax=ax, label='Validation Accuracy (%)')

    _save_and_close(f"{plots_dir}/07b_nn_hyperparameter_surfaces.png")


def plot_nn_hyperparameter_grid_raw(results_list, plots_dir):
    """
    NEW: Show actual measured points without interpolation - honest visualization.
    This complements the interpolated surface plots by showing exactly what was measured.
    """
    optimizers = sorted(set(r['opt'] for r in results_list))

    fig, axes = plt.subplots(1, len(optimizers), figsize=(6 * len(optimizers), 5))
    if len(optimizers) == 1:
        axes = [axes]

    for idx, opt in enumerate(optimizers):
        ax = axes[idx]
        opt_results = [r for r in results_list if r['opt'] == opt]

        # Get unique values
        lrs = sorted(set(r['lr'] for r in opt_results))
        regs = sorted(set(r['reg'] for r in opt_results))

        # Create matrix
        matrix = np.full((len(regs), len(lrs)), np.nan)
        for r in opt_results:
            i = regs.index(r['reg'])
            j = lrs.index(r['lr'])
            matrix[i, j] = r['val_acc'] * 100

        # Plot as heatmap (NO interpolation)
        im = ax.imshow(matrix, cmap='plasma', aspect='auto',
                       vmin=max(np.nanmin(matrix) - 5, 0),
                       vmax=min(np.nanmax(matrix) + 5, 100))

        # Proper tick labels
        ax.set_xticks(range(len(lrs)))
        ax.set_xticklabels([f'{lr:.0e}' for lr in lrs], rotation=45, ha='right')
        ax.set_yticks(range(len(regs)))
        ax.set_yticklabels([f'{reg:.0e}' for reg in regs])

        # Annotate each cell with value
        for i in range(len(regs)):
            for j in range(len(lrs)):
                if not np.isnan(matrix[i, j]):
                    color = 'white' if matrix[i, j] < 70 else 'black'
                    ax.text(j, i, f'{matrix[i, j]:.1f}',
                            ha='center', va='center',
                            color=color, fontweight='bold', fontsize=9)

        # Highlight best cell
        best_i, best_j = np.unravel_index(np.nanargmax(matrix), matrix.shape)
        rect = plt.Rectangle((best_j - 0.5, best_i - 0.5), 1, 1,
                             fill=False, edgecolor='yellow', linewidth=3)
        ax.add_patch(rect)

        ax.set_title(f'{opt.upper()}: Raw Measurements\n(No Interpolation)')
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Regularization Strength')
        plt.colorbar(im, ax=ax, label='Validation Accuracy (%)')

    plt.tight_layout()
    _save_and_close(f"{plots_dir}/07d_nn_hyperparameter_grid_raw.png")


def plot_nn_hidden_size_surfaces(results_list, plots_dir):
    """
    Create surface plots showing Hidden Size vs Learning Rate and vs Regularization
    for the best performing optimizer.
    """
    # Find best optimizer
    best_opt = max(set(r['opt'] for r in results_list),
                   key=lambda opt: np.mean([r['val_acc'] for r in results_list if r['opt'] == opt]))

    opt_results = [r for r in results_list if r['opt'] == best_opt]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Hidden Size vs Learning Rate
    ax1 = axes[0]

    # Average over regularization values
    hidden_lr_results = {}
    for r in opt_results:
        key = (r['hidden'], r['lr'])
        if key not in hidden_lr_results:
            hidden_lr_results[key] = []
        hidden_lr_results[key].append(r['val_acc'] * 100)

    x_data = [k[0] for k in hidden_lr_results.keys()]
    y_data = [k[1] for k in hidden_lr_results.keys()]
    z_data = [np.mean(v) for v in hidden_lr_results.values()]

    x = np.array(x_data)
    y = np.array(y_data)
    z = np.array(z_data)

    log_y = np.log10(y)
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(log_y.min(), log_y.max(), 100)
    Xi, Yi = np.meshgrid(xi, yi)

    Zi = griddata((x, log_y), z, (Xi, Yi), method='linear')

    im1 = ax1.imshow(Zi, extent=(x.min(), x.max(), log_y.min(), log_y.max()),
                     origin='lower', cmap='viridis', aspect='auto')
    ax1.scatter(x, log_y, c=z, cmap='viridis', edgecolor='black', s=100)

    best_idx = np.argmax(z)
    ax1.scatter(x[best_idx], log_y[best_idx], s=200, edgecolor='yellow',
                facecolor='none', linewidth=2)

    unique_lrs = sorted(set(y_data))
    ax1.set_yticks(np.log10(unique_lrs))
    ax1.set_yticklabels([f'{lr:.0e}' for lr in unique_lrs])

    ax1.set_xlabel('Hidden Layer Size')
    ax1.set_ylabel('Learning Rate (log scale)')
    ax1.set_title(f'Hidden Size vs LR (Optimizer: {best_opt})')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(im1, ax=ax1, label='Validation Accuracy (%)')

    # Plot 2: Hidden Size vs Regularization
    ax2 = axes[1]

    # Average over learning rate values
    hidden_reg_results = {}
    for r in opt_results:
        key = (r['hidden'], r['reg'])
        if key not in hidden_reg_results:
            hidden_reg_results[key] = []
        hidden_reg_results[key].append(r['val_acc'] * 100)

    x_data = [k[0] for k in hidden_reg_results.keys()]
    y_data = [k[1] for k in hidden_reg_results.keys()]
    z_data = [np.mean(v) for v in hidden_reg_results.values()]

    x = np.array(x_data)
    y = np.array(y_data)
    z = np.array(z_data)

    log_y = np.log10(y)
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(log_y.min(), log_y.max(), 100)
    Xi, Yi = np.meshgrid(xi, yi)

    Zi = griddata((x, log_y), z, (Xi, Yi), method='linear')

    im2 = ax2.imshow(Zi, extent=(x.min(), x.max(), log_y.min(), log_y.max()),
                     origin='lower', cmap='viridis', aspect='auto')
    ax2.scatter(x, log_y, c=z, cmap='viridis', edgecolor='black', s=100)

    best_idx = np.argmax(z)
    ax2.scatter(x[best_idx], log_y[best_idx], s=200, edgecolor='yellow',
                facecolor='none', linewidth=2)

    unique_regs = sorted(set(y_data))
    ax2.set_yticks(np.log10(unique_regs))
    ax2.set_yticklabels([f'{reg:.0e}' for reg in unique_regs])

    ax2.set_xlabel('Hidden Layer Size')
    ax2.set_ylabel('Regularization Strength (log scale)')
    ax2.set_title(f'Hidden Size vs Reg (Optimizer: {best_opt})')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(im2, ax=ax2, label='Validation Accuracy (%)')

    _save_and_close(f"{plots_dir}/07c_nn_hidden_size_surfaces.png")


def plot_nn_all_surfaces(results_list, plots_dir):
    """
    ENHANCED: Convenience function to create all NN hyperparameter surface plots.
    Now includes both interpolated and raw visualization.

    Args:
        results_list: List of dicts with keys 'hidden', 'lr', 'reg', 'opt', 'val_acc'
        plots_dir: Directory to save plots
    """
    print("Creating Neural Network hyperparameter surface plots...")
    plot_nn_hyperparameter_surfaces(results_list, plots_dir)
    plot_nn_hyperparameter_grid_raw(results_list, plots_dir)  # NEW
    plot_nn_hidden_size_surfaces(results_list, plots_dir)
    print("Neural Network surface plots complete!")


# ============================================================================
# MORE VISUALIZATION!
# ============================================================================

def plot_per_class_accuracy(y_true, y_pred, title, filepath):
    """Plot per-class accuracy as a bar chart with distribution analysis"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    accuracies = []
    support = []
    for i in range(len(CLASS_NAMES)):
        mask = y_true == i
        if np.sum(mask) > 0:
            acc = np.mean(y_pred[mask] == y_true[mask]) * 100
            accuracies.append(acc)
            support.append(np.sum(mask))
        else:
            accuracies.append(0)
            support.append(0)

    # Left plot: Accuracy per class
    bars = ax1.bar(range(len(CLASS_NAMES)), accuracies, color='steelblue', alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Class', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(CLASS_NAMES)))
    ax1.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=np.mean(accuracies), color='red', linestyle='--',
                label=f'Mean: {np.mean(accuracies):.1f}%', linewidth=2)
    ax1.legend()

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)

    # Right plot: Accuracy vs sample count
    ax2.scatter(support, accuracies, s=100, alpha=0.6, c=accuracies, cmap='viridis', edgecolor='black')
    for i, (supp, acc) in enumerate(zip(support, accuracies)):
        ax2.annotate(CLASS_NAMES[i], (supp, acc), fontsize=8,
                     xytext=(5, 5), textcoords='offset points')
    ax2.set_xlabel('Number of Samples', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy vs Class Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    _save_and_close(filepath)


def plot_optimizer_convergence(results_dict, plots_dir):
    """
    Compare convergence speed of different optimizers.

    Args:
        results_dict: Dict with format {optimizer: history_dict}
        plots_dir: Directory to save plots
    """
    plt.figure(figsize=(15, 5))

    # Loss convergence
    plt.subplot(1, 3, 1)
    for opt, hist in results_dict.items():
        if 'loss_history' in hist and len(hist['loss_history']) > 0:
            # Smooth the loss for visibility
            window = max(len(hist['loss_history']) // 50, 10)
            smoothed = np.convolve(hist['loss_history'],
                                   np.ones(window) / window, mode='valid')
            plt.plot(smoothed, label=opt, linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Convergence by Optimizer')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # Training accuracy convergence
    plt.subplot(1, 3, 2)
    for opt, hist in results_dict.items():
        if 'train_acc_history' in hist:
            plt.plot(hist['train_acc_history'], label=opt, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy')
    plt.title('Training Accuracy by Optimizer')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Validation accuracy convergence
    plt.subplot(1, 3, 3)
    for opt, hist in results_dict.items():
        if 'val_acc_history' in hist:
            plt.plot(hist['val_acc_history'], label=opt, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy by Optimizer')
    plt.legend()
    plt.grid(True, alpha=0.3)

    _save_and_close(f"{plots_dir}/13_optimizer_convergence.png")


def plot_overfitting_analysis(history, filepath):
    """Visualize overfitting through train-val gap analysis"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_acc_history']) + 1)

    # Left: Train vs Val accuracy
    ax1.plot(epochs, history['train_acc_history'], 'b-', label='Training', linewidth=2)
    ax1.plot(epochs, history['val_acc_history'], 'r-', label='Validation', linewidth=2)
    ax1.fill_between(epochs, history['train_acc_history'],
                     history['val_acc_history'], alpha=0.3, color='yellow',
                     label='Generalization Gap')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Training vs Validation Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Gap over time
    gap = np.array(history['train_acc_history']) - np.array(history['val_acc_history'])
    colors = ['red' if g > 0.1 else 'orange' if g > 0.05 else 'green' for g in gap]
    ax2.bar(epochs, gap * 100, color=colors, alpha=0.7)
    ax2.axhline(y=5, color='orange', linestyle='--', label='5% threshold')
    ax2.axhline(y=10, color='red', linestyle='--', label='10% threshold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Train-Val Gap (%)')
    ax2.set_title('Overfitting Indicator')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    _save_and_close(filepath)


def plot_nn_heatmap(results_list, plots_dir):
    """Create heatmap for LR vs Reg for best optimizer"""
    # Find best optimizer
    best_opt = max(set(r['opt'] for r in results_list),
                   key=lambda opt: np.mean([r['val_acc'] for r in results_list if r['opt'] == opt]))

    opt_results = [r for r in results_list if r['opt'] == best_opt]

    lrs = sorted(set(r['lr'] for r in opt_results))
    regs = sorted(set(r['reg'] for r in opt_results))

    # Create matrix
    matrix = np.zeros((len(regs), len(lrs)))
    for r in opt_results:
        i = regs.index(r['reg'])
        j = lrs.index(r['lr'])
        matrix[i, j] = r['val_acc'] * 100

    plt.figure(figsize=(10, 8))
    im = plt.imshow(matrix, cmap='viridis', aspect='auto')
    plt.colorbar(im, label='Validation Accuracy (%)')

    plt.xticks(range(len(lrs)), [f'{lr:.0e}' for lr in lrs], rotation=45)
    plt.yticks(range(len(regs)), [f'{reg:.0e}' for reg in regs])
    plt.xlabel('Learning Rate')
    plt.ylabel('Regularization Strength')
    plt.title(f'Hyperparameter Heatmap (Optimizer: {best_opt})')

    # Add text annotations
    for i in range(len(regs)):
        for j in range(len(lrs)):
            plt.text(j, i, f'{matrix[i, j]:.1f}',
                     ha='center', va='center',
                     color='white' if matrix[i, j] < matrix.max() * 0.7 else 'black')

    _save_and_close(f"{plots_dir}/14_nn_heatmap_{best_opt}.png")