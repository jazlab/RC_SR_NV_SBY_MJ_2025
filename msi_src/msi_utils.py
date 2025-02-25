"""
This module contains utility functions for the MSI project.
"""

import os
import torch
import json


# Load the trained model from a checkpoint
def load_checkpoint(model, checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()


def extract_paths(path_dir, type="ortho", require_grad="False"):
    """
    Extracts paths to subdirectories containing the string specified by type

    Parameters:
    - path_dir (str): The directory containing subdirectories to filter.

    Returns:
    - List[Tuple[str, str]]: A list of tuples where each tuple contains the subdirectory path and its corresponding 'id'.
    """
    paths_with_ids = []
    # Iterate over all subdirectories in the path_dir
    for subdir in os.listdir(path_dir):
        subdir_path = os.path.join(path_dir, subdir)

        # Ensure it's a directory
        if os.path.isdir(subdir_path):
            # Check if 'ortho' is in the directory name
            if type in subdir and require_grad in subdir:
                # Extract the id from the directory name
                try:
                    id_value = subdir.split("_id_")[-1]
                except IndexError:
                    id_value = 1
                    # continue  # Skip if 'id' is not found
                # Append the path and id to the results list
                paths_with_ids.append((subdir_path, id_value))

    return paths_with_ids


def load_neural_results(datadir):
    results_neural = {}
    for animal in ["O", "L"]:
        file_name = (
            f"{datadir}/stats_paper/{animal}_fdbk_act_obs_dimensions.json"
        )
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"File not found: {file_name}")
        with open(file_name, "r") as file:
            results_neural[animal] = json.load(file)
    return results_neural
