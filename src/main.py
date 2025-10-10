import os
from joblib import Parallel, delayed
import numpy as np
import random

from config import (
    config,
    SEED,
    DATA_SIZE,
    SUBSAMPLE_TRAIN,
    SUBSAMPLE_VAL,
    PLOTS_DIR,
    CLASS_NAMES,
    KNN_K_VALUES,
    KNN_DISTANCE_METRICS,
    LINEAR_LEARNING_RATES,
    LINEAR_REGULARIZATIONS,
    LINEAR_NUM_ITERS,
    LINEAR_BATCH_SIZE,
    LINEAR_PRINT_EVERY,
    NN_HIDDEN_SIZES,
    NN_LEARNING_RATES,
    NN_REGULARIZATIONS,
    NN_OPTIMIZERS,
    NN_NUM_ITERS,
    NN_BATCH_SIZE,
    NN_PRINT_EVERY,
)
from data import load_data
from k_nearest_neighbor import KNearestNeighbor
from linear_classifier import LinearClassifier
from neural_network import FullyConnectedNN, load_data_nn
from plot_manager import create_plot_manager
from plot import (
    plot_class_distribution,
    plot_knn_validation_and_class_distribution,
    plot_linear_classifier_hyperparameters,
    plot_linear_classifier_learned_weights,
    plot_training_curves,
    plot_confusion_matrix,
    plot_model_comparison,
    visualize_nn_weights,
    visualize_nn_weights_comparison,
    visualize_nn_weights_simple,
    plot_nn_training_loss,
    plot_nn_accuracy_per_epoch,
    plot_nn_hyperparameter_results,
    plot_nn_all_surfaces,
    # ToDo
    plot_per_class_accuracy,
    plot_optimizer_convergence,
    plot_overfitting_analysis,
    plot_nn_heatmap,
)

np.random.seed(SEED)
random.seed(SEED)


def run_knn(X_train, y_train, X_val, y_val, X_test, y_test, plots_dir):
    """Run kNN classifier with hyperparameter tuning"""
    print("\n" + "=" * 60)
    print("PART 1: k-Nearest Neighbors")
    print("=" * 60)

    print("\nRunning k-Nearest Neighbors with parameters:")
    print(f"K values: {KNN_K_VALUES}")
    print(f"Distance metrics: {KNN_DISTANCE_METRICS}")
    print(f"Train samples: {len(y_train)}, Validation samples: {len(y_val)}, Test samples: {len(y_test)}")
    print(f"Feature dimension: {X_train.shape[1]}")

    knn = KNearestNeighbor()
    knn.train(X_train, y_train)

    print("Hyperparameter tuning results:")
    print("k\tL2 accuracy\tL1 accuracy")
    print("-" * 35)

    acc_L2 = []
    acc_L1 = []
    best_accuracy = 0.0
    best_params = {}

    for k in KNN_K_VALUES:
        results = {}
        for metric in KNN_DISTANCE_METRICS:
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
        plots_dir, KNN_K_VALUES, acc_L2, acc_L1, best_params, y_train, y_test
    )

    print(f"\nBest: k={best_params['k']}, {best_params['metric']} ({best_accuracy:.1f}%)")
    final_predictions = knn.predict(X_test, k=best_params['k'], metric=best_params['metric'])
    test_accuracy = np.mean(final_predictions == y_test) * 100
    print(f"Final test accuracy: {test_accuracy:.1f}%")

    plot_confusion_matrix(y_test, final_predictions, "kNN Confusion Matrix", f"{plots_dir}/02_knn_confusion_matrix.png")

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

    return {
        'test_accuracy': test_accuracy,
        'best_params': best_params,
        'predictions': final_predictions
    }


def run_linear_classifiers(X_train, y_train, X_val, y_val, X_test, y_test, plots_dir, num_classes, dim):
    """Run SVM and Softmax classifiers with hyperparameter tuning"""
    print("\n" + "=" * 60)
    print("PART 2: Linear Classifiers")
    print("=" * 60)

    print("\nRunning Linear Classifiers with parameters:")
    print(f"Learning rates: {LINEAR_LEARNING_RATES}")
    print(f"Regularizations: {LINEAR_REGULARIZATIONS}")
    print(f"Iterations: {LINEAR_NUM_ITERS}, Batch size: {LINEAR_BATCH_SIZE}, Print every: {LINEAR_PRINT_EVERY}")
    print(f"Number of classes: {num_classes}, Feature dimension: {dim}")

    def run_grid(loss_type, Xtr, ytr, Xva, yva, lrs, regs, num_iters, batch_size, print_every):
        best_val = -1.0
        best_tuple = None
        best_hist = None

        print(f"\nOptimizing {loss_type} classifier:")
        print("lr\treg\tVal Accuracy")
        print("-" * 35)

        # ----------------------------------
        # Function to run ONE combination
        # ----------------------------------
        def train_one(lr, reg):
            clf = LinearClassifier(
                input_dim=dim,
                num_classes=num_classes,
                loss_type=loss_type
            )

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

            return (lr, reg, val_acc, clf, hist)

        # ----------------------------------
        # Run all combinations in parallel
        # ----------------------------------
        parallel_results = Parallel(n_jobs=-1, verbose=1)(
            delayed(train_one)(lr, reg) for lr in lrs for reg in regs
        )

        results = []
        # Find best
        for lr, reg, val_acc, clf, hist in parallel_results:
            results.append([lr, reg, val_acc])
            print(f"{lr:<7g}\t{reg:<7g}\t{val_acc * 100:.2f}%")

            if val_acc > best_val:
                best_val = val_acc
                best_tuple = (clf, lr, reg)
                best_hist = hist

        # ----------------------------------
        # Final best result
        # ----------------------------------
        clf, lr, reg = best_tuple
        print(f"\nBest {loss_type.capitalize()}: lr={lr}, reg={reg} ({best_val * 100:.2f}%)\n")
        print(results)

        plot_linear_classifier_hyperparameters(plots_dir, results, loss_type)
        return clf, lr, reg, best_hist

    # SVM
    svm_model, best_lr_svm, best_reg_svm, svm_hist = run_grid(
        "svm", X_train, y_train, X_val, y_val,
        LINEAR_LEARNING_RATES, LINEAR_REGULARIZATIONS,
        LINEAR_NUM_ITERS, LINEAR_BATCH_SIZE, LINEAR_PRINT_EVERY
    )
    y_test_pred_svm = svm_model.predict(X_test)
    svm_test_acc = np.mean(y_test_pred_svm == y_test) * 100
    print(f"Final SVM test accuracy: {svm_test_acc:.2f}%")
    plot_linear_classifier_learned_weights(plots_dir, "svm", svm_model.W)

    # Softmax
    softmax_model, best_lr_soft, best_reg_soft, softmax_hist = run_grid(
        "softmax", X_train, y_train, X_val, y_val,
        LINEAR_LEARNING_RATES, LINEAR_REGULARIZATIONS,
        LINEAR_NUM_ITERS, LINEAR_BATCH_SIZE, LINEAR_PRINT_EVERY
    )
    y_test_pred_soft = softmax_model.predict(X_test)
    softmax_test_acc = np.mean(y_test_pred_soft == y_test) * 100
    print(f"Final Softmax test accuracy: {softmax_test_acc:.2f}%")
    plot_linear_classifier_learned_weights(plots_dir, "softmax", softmax_model.W)

    # Training curves
    plot_training_curves(svm_hist, "SVM Training (loss + epoch acc)", f"{plots_dir}/03_svm_training_curves.png")
    plot_training_curves(softmax_hist, "Softmax Training (loss + epoch acc)",
                         f"{plots_dir}/05_softmax_training_curves.png")

    # Confusion matrices
    plot_confusion_matrix(y_test, y_test_pred_svm, "SVM Confusion Matrix", f"{plots_dir}/04_svm_confusion_matrix.png")
    plot_confusion_matrix(y_test, y_test_pred_soft, "Softmax Confusion Matrix",
                          f"{plots_dir}/06_softmax_confusion_matrix.png")

    return {
        'svm': {
            'model': svm_model,
            'test_accuracy': svm_test_acc,
            'best_lr': best_lr_svm,
            'best_reg': best_reg_svm,
            'history': svm_hist,
            'predictions': y_test_pred_svm
        },
        'softmax': {
            'model': softmax_model,
            'test_accuracy': softmax_test_acc,
            'best_lr': best_lr_soft,
            'best_reg': best_reg_soft,
            'history': softmax_hist,
            'predictions': y_test_pred_soft
        }
    }


def run_neural_network(X_train_nn, y_train_nn, X_val_nn, y_val_nn, X_test_nn, y_test_nn, plots_dir, num_classes):
    """Run neural network with hyperparameter tuning"""
    print("\n" + "=" * 60)
    print("PART 3: Neural Networks")
    print("=" * 60)

    print("\nLoading data for Neural Network...")
    print(f"NN data shapes: {X_train_nn.shape}, {X_val_nn.shape}, {X_test_nn.shape}")

    print("\nRunning Neural Network with parameters:")
    print(f"Hidden sizes: {NN_HIDDEN_SIZES}")
    print(f"Learning rates: {NN_LEARNING_RATES}")
    print(f"Regularizations: {NN_REGULARIZATIONS}")
    print(f"Optimizers: {NN_OPTIMIZERS}")
    print(f"Epochs: {NN_NUM_ITERS}, Batch size: {NN_BATCH_SIZE}, Print every: {NN_PRINT_EVERY}")
    print(f"Classes: {num_classes}, Train samples: {len(y_train_nn)}, Val samples: {len(y_val_nn)}, Test samples: {len(y_test_nn)}")

    def run_nn_grid(Xtr, ytr, Xva, yva, hidden_sizes, lrs, regs, optimizers,
                    num_epochs=20, batch_size=64, print_every=10):
        """
        Hyperparameter search for neural network running in parallel.
        """
        best_val = -1.0
        best_tuple = None
        best_hist = None

        print("\nOptimizing Neural Network (PARALLEL):")
        print("Hidden\tLR\tReg\tOptimizer\tVal Accuracy")
        print("-" * 60)

        input_dim = Xtr.shape[1]

        # ----------------------------------
        # Function to train ONE combination
        # ----------------------------------
        def train_one(hidden_size, lr, reg, opt):
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
                num_epochs=num_epochs, batch_size=batch_size,
                optimizer=opt, print_every=print_every
            )

            if hist["val_acc_history"]:
                val_acc = hist["val_acc_history"][-1]
            else:
                val_acc = np.mean(nn.predict(Xva) == yva)

            return (hidden_size, lr, reg, opt, val_acc, nn, hist)

        # ----------------------------------
        # Run all combinations in PARALLEL
        # ----------------------------------
        parallel_results = Parallel(n_jobs=-1, verbose=1)(
            delayed(train_one)(h, lr, reg, opt)
            for h in hidden_sizes
            for lr in lrs
            for reg in regs
            for opt in optimizers
        )

        # ----------------------------------
        # Process results and find best
        # ----------------------------------
        all_results = []
        for h_size, lr, reg, opt, val_acc, nn, hist in parallel_results:
            print(f"{h_size}\t{lr:<7g}\t{reg:<7g}\t{opt}\t\t{val_acc * 100:.2f}%")

            all_results.append({
                'hidden': h_size,
                'lr': lr,
                'reg': reg,
                'opt': opt,
                'val_acc': val_acc
            })

            if val_acc > best_val:
                best_val = val_acc
                best_tuple = (nn, h_size, lr, reg, opt)
                best_hist = hist

        nn, h_size, lr, reg, opt = best_tuple
        print(f"\nBest NN: hidden={h_size}, lr={lr}, reg={reg}, opt={opt} ({best_val * 100:.2f}%)\n")
        return nn, h_size, lr, reg, opt, best_hist, all_results

    # Grid search
    nn_model, best_hidden, best_lr_nn, best_reg_nn, best_opt, nn_hist, nn_results = run_nn_grid(
        X_train_nn, y_train_nn, X_val_nn, y_val_nn,
        NN_HIDDEN_SIZES, NN_LEARNING_RATES, NN_REGULARIZATIONS, NN_OPTIMIZERS,
        NN_NUM_ITERS, NN_BATCH_SIZE, NN_PRINT_EVERY
    )

    y_test_pred_nn = nn_model.predict(X_test_nn)
    nn_test_acc = np.mean(y_test_pred_nn == y_test_nn) * 100
    print(f"Final Neural Network test accuracy: {nn_test_acc:.2f}%")

    plot_nn_hyperparameter_results(nn_results, plots_dir)

    plot_nn_training_loss(nn_hist, f"{plots_dir}/08_nn_training_loss.png")
    plot_nn_accuracy_per_epoch(
        nn_hist,
        f"{plots_dir}/09_nn_training_accuracy.png",
        optimizer=best_opt,
        params={'lr': best_lr_nn, 'reg': best_reg_nn, 'hidden': best_hidden}
    )

    plot_confusion_matrix(y_test_nn, y_test_pred_nn, "Neural Network Confusion Matrix",
                          f"{plots_dir}/10_nn_confusion_matrix.png")

    visualize_nn_weights(nn_model, plots_dir)
    visualize_nn_weights_comparison(nn_model, plots_dir)

    print("\nPer-class accuracy on test set:")
    for i, class_name in enumerate(CLASS_NAMES):
        class_mask = y_test_nn == i
        if np.sum(class_mask) > 0:
            class_acc = np.mean(y_test_pred_nn[class_mask] == y_test_nn[class_mask]) * 100
            print(f"  {class_name:25s}: {class_acc:5.2f}% ({np.sum(class_mask):4d} samples)")

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

    return {
        'model': nn_model,
        'test_accuracy': nn_test_acc,
        'best_params': {
            'hidden_size': best_hidden,
            'learning_rate': best_lr_nn,
            'regularization': best_reg_nn,
            'optimizer': best_opt
        },
        'predictions': y_test_pred_nn,
        'history': nn_hist,
        'all_results': nn_results
    }


def main():
    # Set enabled=True to track the experiment
    USE_EXPERIMENT_TRACKING = True

    if USE_EXPERIMENT_TRACKING:
        # Name that sucker
        # None for auto-generated
        experiment_name = 'nn_optimization_R002'
        plot_manager = create_plot_manager(base_dir=PLOTS_DIR, experiment_name=experiment_name, enabled=True)
        plots_dir = plot_manager.get_plots_dir()

        plot_manager.save_config({
            'seed': SEED,
            'data_size': DATA_SIZE,
            'subsample_train': SUBSAMPLE_TRAIN,
            'subsample_val': SUBSAMPLE_VAL,
            'knn_k_values': KNN_K_VALUES,
            'linear_lrs': LINEAR_LEARNING_RATES,
            'linear_regs': LINEAR_REGULARIZATIONS,
            'nn_hidden_sizes': NN_HIDDEN_SIZES,
            'nn_lrs': NN_LEARNING_RATES,
            'nn_regs': NN_REGULARIZATIONS,
            'nn_optimizers': NN_OPTIMIZERS
        })
    else:
        plots_dir = PLOTS_DIR
        os.makedir(plots_dir, exist_ok=True)
        plot_manager = None

    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(size=DATA_SIZE, subsample_train=SUBSAMPLE_TRAIN)

    plot_class_distribution(PLOTS_DIR, y_train, y_test, y_val)

    print(f"Dataset: {X_train.shape[0]} train, {X_val.shape[0]} val, {X_test.shape[0]} test")
    print(f"Features: {X_train.shape[1]}, Classes: {len(np.unique(y_train))}")
    print()

    num_train, dim = X_train.shape
    num_classes = int(np.max(y_train) + 1)

    results = {}

    # Run kNN
    if config['run_knn']:
        results['knn'] = run_knn(X_train, y_train, X_val, y_val, X_test, y_test, plots_dir)

    # Run Linear Classifiers
    if config['run_linear']:
        linear_results = run_linear_classifiers(
            X_train, y_train, X_val, y_val, X_test, y_test,
            plots_dir, num_classes, dim
        )
        results['svm'] = linear_results['svm']
        results['softmax'] = linear_results['softmax']

    # Run Neural Network
    if config['run_nn']:
        X_train_nn, y_train_nn, X_val_nn, y_val_nn, X_test_nn, y_test_nn = load_data_nn(
            size=DATA_SIZE, subsample_train=SUBSAMPLE_TRAIN, subsample_val=SUBSAMPLE_VAL, seed=SEED
        )
        results['nn'] = run_neural_network(
            X_train_nn, y_train_nn, X_val_nn, y_val_nn, X_test_nn, y_test_nn,
            plots_dir, num_classes
        )

        # Is this check redundant?
        if 'nn' in results and 'all_results' in results['nn']:
            print('\nCreating neural network hyperparameter surface plots...')
            plot_nn_all_surfaces(results['nn']['all_results'], plots_dir)


    # Comparison (only if at least 2 models were run)
    if sum(config.values()) >= 2:
        print("\n" + "=" * 60)
        print("COMPARISON OF ALL MODELS")
        print("=" * 60)

        if 'knn' in results:
            print(f"kNN:            {results['knn']['test_accuracy']:.2f}%")
        if 'svm' in results:
            print(f"SVM:            {results['svm']['test_accuracy']:.2f}%")
        if 'softmax' in results:
            print(f"Softmax:        {results['softmax']['test_accuracy']:.2f}%")
        if 'nn' in results:
            print(f"Neural Network: {results['nn']['test_accuracy']:.2f}%")
        print("=" * 60)

        # Only plot comparison if all models were run
        if all(config.values()):
            plot_model_comparison(
                results['knn']['test_accuracy'],
                results['svm']['test_accuracy'],
                results['softmax']['test_accuracy'],
                results['nn']['test_accuracy'],
                plots_dir
            )

    if USE_EXPERIMENT_TRACKING and plot_manager:
        for model_name, model_results in results.items():
            if isinstance(model_results, dict) and 'test_accuracy' in model_results:
                plot_manager.save_results(model_name, {
                    'test_accuracy': model_results['test_accuracy'],
                    'best_params': model_results.get('best_params', {})
                })
        plot_manager.save_metadata()
        plot_manager.create_comparison_table()

    print("\nDone!")


if __name__ == "__main__":
    main()