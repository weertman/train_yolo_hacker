import os
import yaml
import argparse
from ultralytics import YOLO
import torch
import logging
from logging.handlers import RotatingFileHandler
from typing import Dict, Any

import random
import time

def setup_logging(log_file: str = "trainer.log", log_level: int = logging.INFO):
    """
    Sets up logging configuration for trainer.py.

    :param log_file: Path to the log file.
    :param log_level: Logging level.
    """
    logger = logging.getLogger()
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(processName)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler with rotation
    file_handler = RotatingFileHandler(log_file, maxBytes=10 ** 6, backupCount=5)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def prepare_yaml(dataset_path: str) -> str:
    """
    Prepares the YAML configuration file for the dataset.

    :param dataset_path: Path to the dataset.
    :return: Path to the updated YAML file.
    """
    logging.info(f"Preparing YAML configuration for dataset at '{dataset_path}'.")
    yaml_path = os.path.join(dataset_path, 'data.yaml')
    if not os.path.exists(yaml_path):
        logging.warning(f"YAML configuration file not found at '{yaml_path}'. Creating a new one.")
        # Automatically create data.yaml based on the dataset structure
        labels_dir = os.path.join(dataset_path, 'train', 'labels')
        if not os.path.exists(labels_dir):
            error_msg = f"Labels directory not found at '{labels_dir}'. Cannot create data.yaml."
            logging.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Determine the number of classes and class names
        class_names = set()
        for label_file in os.listdir(labels_dir):
            if label_file.endswith('.txt'):
                with open(os.path.join(labels_dir, label_file), 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_id = parts[0]
                            class_names.add(class_id)

        # Sort class IDs numerically and create class names (modify as per your actual class names)
        try:
            sorted_class_ids = sorted(class_names, key=lambda x: int(x))
        except ValueError:
            logging.error("Class IDs are not integers. Please ensure labels are correctly formatted.")
            raise

        data = {
            'train': os.path.join(dataset_path, 'train', 'images'),
            'val': os.path.join(dataset_path, 'valid', 'images'),
            'test': os.path.join(dataset_path, 'test', 'images'),
            'nc': len(sorted_class_ids),
            'names': [f"class_{cls}" for cls in sorted_class_ids]  # Replace with actual class names if available
        }

        # Write the YAML file
        with open(yaml_path, 'w') as file:
            yaml.dump(data, file)

        logging.info(f"YAML configuration created at '{yaml_path}'.")
    else:
        logging.info(f"YAML configuration file found at '{yaml_path}'. Updating paths.")

        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)

        data['train'] = os.path.join(dataset_path, 'train', 'images')
        data['val'] = os.path.join(dataset_path, 'valid', 'images')
        data['test'] = os.path.join(dataset_path, 'test', 'images')

        # Optionally, update 'nc' and 'names' if necessary
        # This step assumes that 'names' and 'nc' are already correctly set
        # If not, you may need to update them similarly to the creation step

        with open(yaml_path, 'w') as file:
            yaml.dump(data, file)

        logging.info(f"YAML configuration updated at '{yaml_path}'.")

    return yaml_path  # Ensure the path is returned


def verify_dataset(dataset_path: str):
    """
    Verifies the structure of the downloaded dataset.

    :param dataset_path: Path to the dataset.
    """
    logging.info(f"Verifying dataset structure at '{dataset_path}'.")
    required_subdirs = [
        'train/images',
        'train/labels',
        'valid/images',
        'valid/labels',
        'test/images',
        'test/labels'
    ]
    missing_dirs = []
    for subdir in required_subdirs:
        full_path = os.path.join(dataset_path, subdir)
        if not os.path.exists(full_path):
            missing_dirs.append(subdir)

    if missing_dirs:
        error_msg = f"Dataset directory structure is incorrect. Missing: {', '.join(missing_dirs)}."
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)

    logging.info("Dataset structure verified successfully.")


def train_model(config: Dict[str, Any]) -> YOLO:
    """
    Trains the YOLO model based on the provided configuration.

    :param config: Dictionary containing training parameters.
    :return: Trained YOLO model.
    """
    logging.info(f"Initializing YOLO model: {config['model']}")
    try:
        time.sleep(random.randint(1,4))
        model = YOLO(config['model'])
        print("Model loaded successfully.")
    except Exception as e:
        ## wait a random period of time 2-10 seconds
        time.sleep(random.randint(2, 10))
        ## try again
        try:
            model = YOLO(config['model'])
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model second time: {e}")
        print(f"Error loading model: {e}")
    logging.info("Starting training")
    results = model.train(
        data=config['yaml_path'],
        epochs=config['epochs'],
        imgsz=config['imgsz'],
        patience=config['patience'],
        batch=config['batch'],
        device=torch.device(config['device']),
        close_mosaic=config['close_mosaic'],
        mosaic=config['mosaic'],
    )
    logging.info("Training completed")
    return model


def evaluate_model(model: YOLO, dataset_path: str):
    """
    Evaluates the trained YOLO model.

    :param model: Trained YOLO model.
    :param dataset_path: Path to the dataset.
    """
    logging.info("Starting model evaluation")
    model.val()
    test_images_dir = os.path.join(dataset_path, 'test', 'images')
    results = model(test_images_dir)
    logging.info(f"Training and evaluation completed. Results saved in {model.trainer.save_dir}")


def hyperparameter_tuning(base_config: Dict[str, Any], param_grid: Dict[str, list]) -> YOLO:
    """
    Performs hyperparameter tuning to find the best model.

    :param base_config: Base configuration dictionary.
    :param param_grid: Dictionary of hyperparameters to tune.
    :return: Best performing YOLO model.
    """
    logging.info("Starting hyperparameter tuning")
    best_model = None
    best_performance = float('inf')  # Assuming lower is better, change if needed

    for param, values in param_grid.items():
        for value in values:
            current_config = base_config.copy()
            current_config[param] = value
            logging.info(f"Training with {param}={value}")
            model = train_model(current_config)
            performance = model.metrics.get('val/loss', float('inf'))  # Change metric if needed
            logging.info(f"Performance with {param}={value}: {performance}")

            if performance < best_performance:
                best_performance = performance
                best_model = model
                logging.info(f"New best model found with {param}={value}")

    logging.info("Hyperparameter tuning completed")
    return best_model


def main(args):
    try:
        # Verify that the dataset path is provided
        if not hasattr(args, 'dataset_path') or not args.dataset_path:
            error_msg = "Dataset path is not provided. Please ensure that 'dataset_path' is specified."
            logging.error(error_msg)
            raise ValueError(error_msg)

        dataset_path = args.dataset_path

        # Prepare YAML and verify dataset structure
        yaml_path = prepare_yaml(dataset_path)
        verify_dataset(dataset_path)

        config = {
            'model': args.model,
            'yaml_path': yaml_path,
            'epochs': args.epochs,
            'imgsz': args.imgsz,
            'patience': args.patience,
            'batch': args.batch,
            'device': args.device,
            'close_mosaic': args.close_mosaic,
            'mosaic': args.mosaic,
        }

        if args.tune:
            param_grid = {
                'batch': [8, 16, 32],
                'imgsz': [480, 640],
                'mosaic': [0.5, 0.75, 1.0],
            }
            best_model = hyperparameter_tuning(config, param_grid)
        else:
            best_model = train_model(config)

        evaluate_model(best_model, dataset_path)
    except Exception as e:
        logging.error(f"An error occurred in main: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    setup_logging(log_file="trainer.log")
    logging.info("Starting trainer.py")

    parser = argparse.ArgumentParser(description="YOLO Training Script")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the downloaded dataset")
    parser.add_argument('--model', type=str, default='yolov8s-seg.pt', help="YOLO model to use")
    parser.add_argument('--epochs', type=int, default=150, help="Number of training epochs")
    parser.add_argument('--imgsz', type=int, default=640, help="Input image size")
    parser.add_argument('--patience', type=int, default=150, help="Early stopping patience")
    parser.add_argument('--batch', type=int, default=32, help="Batch size")
    parser.add_argument('--device', type=str, default='0', help="Device to use (e.g., 'cpu', '0', '0,1')")
    parser.add_argument('--close_mosaic', type=int, default=10, help="Close mosaic")
    parser.add_argument('--mosaic', type=float, default=0.75, help="Mosaic")
    parser.add_argument('--tune', action='store_true', help="Perform hyperparameter tuning")
    args = parser.parse_args()

    main(args)
    logging.info("trainer.py finished successfully")
