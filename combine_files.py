import os
import logging


def combine_python_files(
        project_root,
        output_file_path,
        exclude_dirs=['dev'],
        file_extension='.py'
):
    """
    Combines all Python files in the project directory into a single file with metadata.

    Args:
        project_root (str): The root directory of the project.
        output_file_path (str): The path to the output combined Python file.
        exclude_dirs (list): List of directory names to exclude from the search.
        file_extension (str): The file extension to look for (default is '.py').

    Returns:
        None
    """
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for root, dirs, files in os.walk(project_root):
            # Modify dirs in-place to exclude certain directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                if file.endswith(file_extension):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, project_root)

                    # Add metadata as comments
                    outfile.write(f"# ======= Start of {relative_path} =======\n")

                    try:
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            content = infile.read()
                            outfile.write(content)
                    except Exception as e:
                        logging.error(f"Failed to read {file_path}: {e}")
                        outfile.write(f"# Error reading file: {e}\n")

                    outfile.write(f"\n# ======= End of {relative_path} =======\n\n")

    logging.info(f"All Python files have been combined into {output_file_path}")
