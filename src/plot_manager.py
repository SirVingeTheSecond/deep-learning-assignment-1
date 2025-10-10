import json
import os
from datetime import datetime
import matplotlib.pyplot as plt


class PlotManager:
    """
    Optional plot management system for organizing plots across experiments.
    Completely backward compatible - existing code works without modification.
    """

    def __init__(self, base_dir="plots", experiment_name=None, enabled=True):
        """
        Initialize the plot manager.

        Args:
            base_dir: Base directory for all plots
            experiment_name: Name of this experiment (auto-generated if None)
            enabled: If False, acts as pass-through (for backward compatibility)
        """
        self.enabled = enabled
        self.base_dir = base_dir

        if not enabled:
            # Pass-through mode - just use base_dir
            self.experiment_dir = base_dir
            os.makedirs(self.experiment_dir, exist_ok=True)
            return

        # Full experiment tracking mode
        if experiment_name is None:
            experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.experiment_name = experiment_name
        self.experiment_dir = os.path.join(base_dir, experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)

        self.metadata = {
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'config': {},
            'plots': [],
            'results': {}
        }

    def get_plots_dir(self):
        """Get the plots directory path"""
        return self.experiment_dir

    def save_plot(self, fig, name, category=None, params=None, dpi=300):
        """
        Save a plot with optional metadata tracking.

        Args:
            fig: matplotlib figure object
            name: base name for the file
            category: optional category subdirectory (e.g., 'knn', 'linear', 'neural_network')
            params: optional dict of parameters used to generate this plot
            dpi: resolution for saved image

        Returns:
            filepath: path where plot was saved
        """
        if not self.enabled:
            # Simple mode - just save to plots_dir
            filepath = os.path.join(self.experiment_dir, name)
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            return filepath

        # Full tracking mode
        if category:
            category_dir = os.path.join(self.experiment_dir, category)
            os.makedirs(category_dir, exist_ok=True)
            save_dir = category_dir
        else:
            save_dir = self.experiment_dir

        # Generate filename with parameters if provided
        if params:
            param_str = "_".join([f"{k}{v}" for k, v in sorted(params.items())])
            filename = f"{name}_{param_str}.png"
        else:
            filename = name if name.endswith('.png') else f"{name}.png"

        filepath = os.path.join(save_dir, filename)
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

        # Track in metadata
        self.metadata['plots'].append({
            'name': name,
            'category': category,
            'filepath': filepath,
            'params': params,
            'timestamp': datetime.now().isoformat()
        })

        return filepath

    def save_config(self, config_dict):
        """Save experiment configuration"""
        if self.enabled:
            self.metadata['config'] = config_dict

    def save_results(self, model_name, results_dict):
        """Save model results"""
        if self.enabled:
            self.metadata['results'][model_name] = results_dict

    def save_metadata(self):
        """Write metadata to disk"""
        if self.enabled:
            filepath = os.path.join(self.experiment_dir, 'metadata.json')
            with open(filepath, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            print(f"Saved experiment metadata to {filepath}")

    def create_comparison_table(self):
        """Generate a markdown comparison table of all models"""
        if not self.enabled or not self.metadata['results']:
            return

        lines = ["# Experiment Results\n"]
        lines.append(f"**Experiment:** {self.experiment_name}\n")
        lines.append(f"**Date:** {self.metadata['timestamp']}\n\n")
        lines.append("## Model Comparison\n")
        lines.append("| Model | Test Accuracy | Best Hyperparameters |")
        lines.append("|-------|---------------|---------------------|")

        for model_name, results in self.metadata['results'].items():
            acc = results.get('test_accuracy', 'N/A')
            if isinstance(acc, float):
                acc = f"{acc:.2f}%"

            params = results.get('best_params', {})
            param_str = ', '.join([f"{k}={v}" for k, v in params.items()])

            lines.append(f"| {model_name} | {acc} | {param_str} |")

        filepath = os.path.join(self.experiment_dir, 'results_summary.md')
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))

        print(f"Saved results summary to {filepath}")


# Convenience function for backward compatibility
def create_plot_manager(base_dir="plots", experiment_name=None, enabled=False):
    """
    Factory function to create a PlotManager.
    Set enabled=True to use experiment tracking, False for simple mode.
    """
    return PlotManager(base_dir=base_dir, experiment_name=experiment_name, enabled=enabled)