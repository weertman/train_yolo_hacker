import os
import sys
import yaml
import logging
from logging.handlers import RotatingFileHandler
from roboflow import Roboflow
import argparse
import shutil


def setup_logging(log_file: str = "dataset_downloader.log", log_level: int = logging.INFO):
    """
    Sets up logging configuration.
    """
    logger = logging.getLogger()
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
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
    If data.yaml does not exist, it creates one based on the dataset structure.

    :param dataset_path: Path to the dataset.
    :return: Path to the updated or created YAML file.
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

    return yaml_path


def is_dataset_complete(dataset_path: str) -> bool:
    """
    Checks if the dataset directory contains all required subdirectories.

    :param dataset_path: Path to the dataset.
    :return: True if complete, False otherwise.
    """
    required_subdirs = [
        'train/images',
        'train/labels',
        'valid/images',
        'valid/labels',
        'test/images',
        'test/labels'
    ]
    for subdir in required_subdirs:
        full_path = os.path.join(dataset_path, subdir)
        if not os.path.exists(full_path):
            logging.warning(f"Missing required subdirectory: '{subdir}'")
            return False
    return True


def download_dataset(api_key: str, workspace: str, project: str, version: int, dataset_name: str) -> str:
    logging.info(f"Starting download for dataset '{dataset_name}' from Roboflow.")
    logging.info(f"api_key: {api_key}")
    logging.info(f"workspace: {workspace}")
    logging.info(f"project: {project}")
    logging.info(f"version: {version}")
    logging.info(f"dataset_name: {dataset_name}")

    rf = Roboflow(api_key=api_key)
    project_rf = rf.workspace(workspace).project(project)
    version_rf = project_rf.version(version)
    dataset_path = os.path.abspath(os.path.join('datasets', dataset_name))

    if os.path.exists(dataset_path):
        if is_dataset_complete(dataset_path):
            logging.info(
                f"Dataset '{dataset_name}' already exists at '{dataset_path}' and is complete. Skipping download.")
        else:
            logging.warning(f"Dataset '{dataset_name}' directory exists but is incomplete. Re-downloading.")
            try:
                shutil.rmtree(dataset_path)
                logging.info(f"Deleted incomplete dataset directory '{dataset_path}'.")
            except Exception as e:
                logging.error(f"Failed to delete incomplete dataset directory '{dataset_path}': {e}")
                raise e
            # Proceed to download after deletion

    os.makedirs(dataset_path, exist_ok=True)

    try:
        logging.info(f"Downloading dataset '{dataset_name}' to '{dataset_path}'...")
        dataset = version_rf.download("yolov11", location=dataset_path)
        logging.info(f"Dataset '{dataset_name}' downloaded successfully to '{dataset_path}'.")
    except Exception as e:
        logging.error(f"Failed to download dataset: {e}")
        raise

    return dataset_path



def verify_dataset_structure(dataset_path: str):
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


def main(args):
    setup_logging(log_file=args.log_file)
    logging.info("Starting dataset_downloader.py")

    try:
        dataset_path = download_dataset(
            api_key=args.api_key,
            workspace=args.workspace,
            project=args.project,
            version=args.version,
            dataset_name=args.dataset_name
        )
        prepare_yaml(dataset_path)
        verify_dataset_structure(dataset_path)
    except Exception as e:
        logging.error(f"An error occurred during dataset download: {e}", exc_info=True)
        sys.exit(1)

    logging.info("Dataset download and preparation completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Roboflow Dataset Downloader")
    parser.add_argument('--api_key', type=str, required=True, help="Roboflow API key")
    parser.add_argument('--workspace', type=str, required=True, help="Roboflow workspace name")
    parser.add_argument('--project', type=str, required=True, help="Roboflow project name")
    parser.add_argument('--version', type=int, required=True, help="Dataset version number")
    parser.add_argument('--dataset_name', type=str, required=True, help="Name for the downloaded dataset directory")
    parser.add_argument('--log_file', type=str, default="dataset_downloader.log", help="Path to the log file")
    args = parser.parse_args()

    main(args)
