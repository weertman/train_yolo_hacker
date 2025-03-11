import os
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from collections import defaultdict
import shutil  # Import shutil for copying files

def get_run_num(run_dir):
    run_num = run_dir.split('train')[-1]
    if len(run_num) == 0:
        return 0
    return int(run_num)

if __name__ == '__main__':
    root_run_dir = Path(r'D:\trainYolo\src\runs\segment')
    run_dirs = [d for d in root_run_dir.iterdir() if d.is_dir()]
    run_dirs = [d for d in run_dirs if (d / 'args.yaml').exists()]
    run_dirs = sorted(run_dirs, key=lambda x: get_run_num(str(x)))

    # List of metrics to plot
    metrics_to_plot = [
        'metrics/precision(M)',
        'metrics/recall(M)',
        'metrics/mAP50(M)',
        'metrics/mAP50-95(M)',
        'metrics/precision(B)',
        'metrics/recall(B)',
        'metrics/mAP50(B)',
        'metrics/mAP50-95(B)',
        'train/box_loss',
        'train/obj_loss',
        'train/seg_loss',
        'val/box_loss',
        'val/obj_loss',
        'val/seg_loss'
    ]

    # Collect arguments from each run
    run_args_list = []

    for run in run_dirs:
        args_path = run / 'args.yaml'
        with open(args_path, 'r') as file:
            args = yaml.safe_load(file)
        run_args_list.append(args)

    # Identify differing arguments
    all_keys = set().union(*(args.keys() for args in run_args_list))
    args_values = defaultdict(set)

    for args in run_args_list:
        for key in all_keys:
            args_values[key].add(args.get(key, None))

    differing_args = {key for key, values in args_values.items() if len(values) > 1}

    print("Differing arguments between runs:")
    for key in sorted(differing_args):
        print(f"- {key}")

    # Create a summary DataFrame to collect final metrics
    summary_data = []

    for metric in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        data_plotted = False  # Flag to check if data was plotted

        for idx, run in enumerate(run_dirs):
            args = run_args_list[idx]
            model_type = args['model']
            model_type = Path(model_type).stem

            # Extract only the differing arguments
            differing_args_values = {key: args.get(key, 'N/A') for key in differing_args}

            # Build label using differing arguments
            label_parts = [f"{key}={value}" for key, value in differing_args_values.items()]
            label = f"{model_type} - " + ", ".join(label_parts)

            path_results = run / 'results.csv'
            if not path_results.exists():
                continue  # Skip if results.csv does not exist

            results = pd.read_csv(path_results)
            if metric not in results.columns:
                continue  # Skip if metric is not in the results

            metric_values = results[metric]
            if metric_values.isnull().all():
                continue  # Skip if all values are NaN

            max_metric = metric_values.max()
            idx_max_metric = metric_values.idxmax()

            # Update label to include max metric value
            label += f"\nmax {metric}: {max_metric:.4f} @ epoch {idx_max_metric}"

            ax.plot(metric_values, label=label)
            data_plotted = True  # Set flag to True if data is plotted

            # Get the path to the best model file
            best_model_path = run / 'weights' / 'best.pt'
            if not best_model_path.exists():
                print(f"Best model file not found for run {run}.")
                best_model_path = None

            # Collect final metrics for the summary
            if metric == 'metrics/mAP50-95(M)':
                summary_data.append({
                    'Model': model_type,
                    **differing_args_values,
                    'Max mAP50-95(M)': max_metric,
                    'Epoch at Max mAP50-95(M)': idx_max_metric,
                    'Best Model Path': str(best_model_path) if best_model_path else None
                })

        if data_plotted:
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} over Epochs')
            ax.legend(fontsize='small', loc='best')
            plt.tight_layout()
            # Save the plot
            plot_filename = f'{metric.replace("/", "_")}_comparison.png'
            plt.savefig(root_run_dir / plot_filename)
            plt.close()
        else:
            plt.close()
            print(f"No data available for metric '{metric}'. Plot not generated.")

    # Create a summary DataFrame and save it
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = root_run_dir / 'summary_metrics.csv'
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"Summary of final metrics saved to '{summary_csv_path}'.")

        # Find the best model for each model type
        best_models_dir = root_run_dir / 'best_models'
        best_models_dir.mkdir(exist_ok=True)

        grouped = summary_df.groupby('Model')
        for model_type, group in grouped:
            # Find the run with the highest Max mAP50-95(M)
            best_run = group.loc[group['Max mAP50-95(M)'].idxmax()]
            best_model_path = best_run['Best Model Path']
            if best_model_path and os.path.exists(best_model_path):
                # Copy the best model file to the best_models_dir
                dest_path = best_models_dir / f"{model_type}_best.pt"
                shutil.copy(best_model_path, dest_path)
                print(f"Copied best model for {model_type} to {dest_path}")
            else:
                print(f"No best model file found for {model_type}")
    else:
        print("No data collected for summary.")

