import sys
import os
import torch
import multiprocessing
import argparse
import logging
from logging.handlers import RotatingFileHandler
import subprocess
import urllib.request
import uuid

# Add the directory containing the main script to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Import the main function from the trainer script
from trainer import main as trainer_main
from combine_files import combine_python_files

def setup_logging(log_file: str = None, log_level: int = logging.INFO):
    """
    Sets up logging configuration.
    """
    logger = logging.getLogger()
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(processName)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            # File handler with rotation
            file_handler = RotatingFileHandler(log_file, maxBytes=10**6, backupCount=5)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

def run_dataset_downloader(args):
    """
    Runs the dataset_downloader.py script to download and prepare datasets.
    """
    logging.info("Starting dataset download process.")
    try:
        subprocess.check_call([
            sys.executable,  # Path to the Python interpreter
            os.path.join(script_dir, 'dataset_downloader.py'),
            '--api_key', args.api_key,
            '--workspace', args.workspace,
            '--project', args.project,
            '--version', str(args.version),
            '--dataset_name', args.dataset_name,
            '--log_file', 'dataset_downloader.log'
        ])
        logging.info("Dataset downloaded and prepared successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Dataset download failed with error: {e}")
        sys.exit(1)

def run_experiment(experiment_config, gpu_id, experiment_id):
    """
    Executes a single training experiment.
    """
    unique_id = uuid.uuid4().hex
    log_file = f"trainer_experiment_{experiment_id}_{unique_id}.log"
    setup_logging(log_file=log_file)
    logging.info(f"Running experiment {experiment_id} on GPU {gpu_id}")

    # Update device in experiment config
    experiment_config['device'] = str(gpu_id)
    experiment_config.setdefault('tune', False)

    # Ensure the dataset_path is passed to the trainer
    dataset_path = os.path.abspath(os.path.join('datasets', experiment_config['dataset_name']))
    experiment_config['dataset_path'] = dataset_path

    # Remove dataset-related parameters as they're handled by the downloader
    for key in ['api_key', 'workspace', 'project', 'version', 'dataset_name']:
        experiment_config.pop(key, None)

    # Convert dict to argparse.Namespace
    args_namespace = argparse.Namespace(**experiment_config)
    try:
        trainer_main(args_namespace)
        logging.info(f"Experiment {experiment_id} on GPU {gpu_id} completed successfully")
    except Exception as e:
        logging.error(f"Experiment {experiment_id} on GPU {gpu_id} failed with error: {e}", exc_info=True)

def main():
    # Set up logging
    setup_logging(log_file="configurator.log")

    logging.info("Starting configurator.py")

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))

    ## combine_files.py
    logging.info("Combining Python files for GPT to read")
    combine_python_files(
        project_root=PROJECT_ROOT,
        output_file_path=os.path.join(PROJECT_ROOT, 'dev', 'combined_files.py'),
        exclude_dirs=['dev'],
        file_extension='.py'
    )


    # Define default arguments
    default_args = {
        "api_key": "3DMRGtyRTVJorWA7u61H",                # Replace with your actual Roboflow API key
        "workspace": "sunflowerstarlab",                 # Replace with your workspace name
        "project": "sunflowerstarseg",                   # Replace with your project name
        "version": 13,                                    # Replace with your dataset version number
        "dataset_name": "SunflowerStarSeg_13_yolov11"    # Replace with your desired dataset directory name
    }

    # Convert default_args dict to argparse.Namespace
    args_namespace = argparse.Namespace(**default_args)

    # Step 1: Download and prepare the dataset
    run_dataset_downloader(args_namespace)

    # Define the directory to save models
    model_dir = r'D:\trainYolo\src\runs\segment\starSegPreTrainingRun\best_models'
    os.makedirs(model_dir, exist_ok=True)

    # Define experiment variations with absolute paths
    models = [
        #os.path.join(model_dir, 'yolo11n-seg_best.pt'),
        os.path.join(model_dir, 'yolo11s-seg_best.pt'),
        os.path.join(model_dir, 'yolo11m-seg_best.pt'),
        os.path.join(model_dir, 'yolo11l-seg_best.pt'),
        #os.path.join(model_dir, 'yolo11x-seg_best.pt')
    ]
    batch_sizes = {
        #os.path.join(model_dir, 'yolo11n-seg_best.pt'): [12],
        os.path.join(model_dir, 'yolo11s-seg_best.pt'): [12],
        os.path.join(model_dir, 'yolo11m-seg_best.pt'): [12],
        os.path.join(model_dir, 'yolo11l-seg_best.pt'): [12],
        #os.path.join(model_dir, 'yolo11x-seg_best.pt'): [12]
    }
    mosaic_values = [0.5]

    # Generate experiments
    base_config = {
        "model": "",             # To be updated per experiment
        "epochs": 200,
        "imgsz": 720,
        "patience": 50,
        "close_mosaic": 10,
        "dataset_name": default_args['dataset_name'],  # Pass dataset name for path resolution
    }

    experiments = []
    experiment_id = 0
    for model in models:
        for batch in batch_sizes[model]:
            for mosaic in mosaic_values:
                experiment = base_config.copy()
                experiment.update({
                    "model": model,
                    "batch": batch,
                    "mosaic": mosaic,
                    "tune": False  # Ensure 'tune' is present
                })
                experiments.append((experiment, 'cuda:1', experiment_id))  # Assign GPU IDs cyclically
                experiment_id += 1
                ## report the experiment settings
                logging.info(f"Experiment {experiment_id} settings: {experiment}")

    logging.info(f"Total number of experiments to run: {len(experiments)}")

    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    logging.info(f"Number of available GPUs: {num_gpus}")

    if num_gpus == 0:
        logging.warning("No GPUs available. Running on CPU.")
        num_gpus = 1  # Fallback to CPU

    ## run the experiments linearly for now
    for experiment, gpu_id, experiment_id in experiments:
        ## report the info for the experiment
        logging.info(f"Running experiment {experiment_id} on GPU {gpu_id}")
        logging.info(f"Experiment config: {experiment}")
        logging.info(f"Experiment ID: {experiment_id}")

        run_experiment(experiment, gpu_id, experiment_id)

    logging.info("All experiments completed")

if __name__ == "__main__":
    main()
