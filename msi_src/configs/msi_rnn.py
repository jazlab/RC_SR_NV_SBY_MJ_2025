"""Configurations to train a MultiRNN model on MSI task.

See ../models/multi_rnn.py for model details.
"""

from models import multi_rnn
import msi_task
import torch
import trainer


def model_config():

    config = {
        "constructor": multi_rnn.MultiRNN,
        "kwargs": {
            "input_feature_len": 3,
            "hidden_size": 200,
            "activity_decay": 0.2,
        },
    }

    return config


def get_config():
    """Get config for main.py."""

    config = {
        "constructor": trainer.Trainer,
        "kwargs": {
            "model": {
                "constructor": multi_rnn.MultiRNN,
                "kwargs": {
                    "input_feature_len": 3,
                    "hidden_size": 200,
                    "activity_decay": 0.2,
                    "require_input_grad": False,
                    "require_output_grad": False,
                },
            },
            "optim_config": {
                "optimizer": torch.optim.Adam,
                "kwargs": {
                    "lr": 1e-4,
                },
            },
            "task": {
                "constructor": msi_task.MSI,
                "kwargs": {
                    "timesteps": 500,
                    "model_type": "ortho",
                    "integration_factor": 0.5,
                    "mean_choices": [-0.5, 0.5],
                    "value_std": 0.1,
                    "iti": 5,
                },
            },
            "batch_size": 16,
            "iterations": int(1e5) + 1,
            "scalar_eval_every": 50,
            "image_eval_every": 200,
            "snapshot_every": 5000,
            "id": 100,
        },
    }

    return config
