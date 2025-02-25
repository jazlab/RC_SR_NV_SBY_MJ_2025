"""
Demo the MSI task in ./msi_task.py.
Output will be a plot of the stimulus and desired response for 
FLAGS.num_trials task trials.
Options: model_type = 'paral' or 'ortho' where paral is parallel (H1) 
and ortho is orthogonal (H2) model.
"""

from absl import app
from absl import flags
from matplotlib import pyplot as plt
import numpy as np

import msi_task

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_trials", 1, "Number of task trials to plot.")


def main(_):
    """Plot stimulus and desired response for FLAGS.num_trials task trials."""

    task = msi_task.MSI(
        timesteps=100,
        model_type="paral",
        integration_factor=0.5,
        mean_choices=[-0.5, 0.5],
        value_std=0.1,
        iti=2,
    )

    for _ in range(FLAGS.num_trials):
        data = task()
        _, ax = plt.subplots(2, 1, figsize=(6, 6))
        ax[0].plot(np.squeeze(data["inputs"])[:, 0], "b")
        ax[0].plot(np.squeeze(data["inputs"])[:, 1], "r")
        ax[0].plot(np.squeeze(data["inputs"])[:, 2] + 0.1, "k")
        ax[0].set_xlim([0, len(data["labels"])])
        ax[0].set_ylabel("Stimulus")
        index_array = np.arange(len(data["labels"]))
        ax[1].plot(index_array, np.squeeze(data["labels"]))
        ax[1].set_xlim([0, len(data["labels"])])
        ax[1].set_ylabel("Desired Response")
        ax[1].set_xlabel("Time")

    plt.show()


if __name__ == "__main__":
    app.run(main)
