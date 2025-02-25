"""Trainer for model on msi task.

The scalar, image, and figures logged during training can be visualized in
tensorboard by running the following command:
$ python3 tensorboard --log_dir=logs/$run_number/tensorboard
"""

import logging
import numpy as np
import os
from python_utils.logging import scalar_writer as scalar_writer_lib
import torch
from torch.utils import tensorboard


class Trainer(object):
    """Trains a model on any task.

    The self.__call__() method performs a training loop and logs scalars, images,
    figures, and snapshots.
    """

    def __init__(
        self,
        model,
        optim_config,
        task,
        batch_size,
        iterations,
        scalar_eval_every,
        image_eval_every,
        snapshot_every,
        id,
    ):
        """Constructor.

        Args:
            model: Instance of a model subclassed from torch.nn.Module. Should
                have 'train_step', 'scalars', 'images', and 'figures' methods
                and 'scalar_keys' property.
            optim_config: Optimizer config. Must have keys 'optimizer', which
                should be a torch.optim optimizers, and 'kwargs', which must
                include all arguments except the parameters.
            task: Instance of task. Should be callable, returning a data
                dictionary, and should have a 'data_keys' property defining the
                keys in the data dictionary.
            batch_size: Int. Batch size.
            iterations: Int. Number of training steps.
            scalar_eval_every: Int. Scalar evaluation period.
            image_eval_every: Int. Evaluation period for images and figures.
            snapshot_every: Int. Snapshot period.
            id: Int. Unique identifier for this run.
        """
        # set random seed to be id
        torch.manual_seed(id)
        self._model = model
        self._optimizer = optim_config["optimizer"](
            model.parameters(), **optim_config["kwargs"]
        )
        self._task = task
        self._batch_size = batch_size
        self._iterations = iterations
        self._scalar_eval_every = scalar_eval_every
        self._image_eval_every = image_eval_every
        self._snapshot_every = snapshot_every

    def get_data_batch(self, batch_size=None):
        """Get batch of data.

        Args:
            batch_size: Int or None. Batch size. If None, defaults to
                self._batch_size.

        Returns:
            torch_data_batch: Dictionary of tensors, each batch-major, namely
                with first dimension batch_size.
        """
        if batch_size is None:
            batch_size = self._batch_size
        data_dicts = [self._task() for _ in range(batch_size)]
        data_batch = {
            k: np.stack([d[k] for d in data_dicts])
            for k in self._task.data_keys
        }
        torch_data_batch = {
            k: torch.from_numpy(v) for k, v in data_batch.items()
        }
        return torch_data_batch

    def train_step(self, data):
        """Take a training step of the model on a data batch."""
        self._optimizer.zero_grad()
        loss_terms = self._model.loss_terms(self._model.forward(data))
        loss = sum(loss_terms.values())
        loss.backward()
        self._optimizer.step()

    def __call__(self, log_dir):
        """Run training loop, logging scalars, images, figures, and snapshots.

        Args:
            log_dir: String. Path to logging directory.
        """

        snapshot_dir = os.path.join(log_dir, "snapshots")
        os.makedirs(snapshot_dir)

        scalars_filename = os.path.join(log_dir, "scalars.csv")
        scalar_writer = scalar_writer_lib.ScalarWriter(
            scalars_filename, self._model.scalar_keys + ("step",)
        )

        summary_dir = os.path.join(log_dir, "tensorboard")
        summary_writer = tensorboard.SummaryWriter(log_dir=summary_dir)

        for step in range(self._iterations):

            data_batch = self.get_data_batch()
            self.train_step(data_batch)

            if step % self._scalar_eval_every == 0:
                logging.info("Logging scalars.")
                scalars = self._model.scalars(data_batch)
                scalars["step"] = step
                logging.info("Step: {} of {}".format(step, self._iterations))
                logging.info(scalars)
                scalar_writer.write(scalars)
                for k, v in scalars.items():
                    summary_writer.add_scalar(k, v, global_step=step)

            if step % self._image_eval_every == 0:
                logging.info("Logging figures.")
                figures = self._model.figures(data_batch)
                for k, v in figures.items():
                    summary_writer.add_figure(k, v, global_step=step)

            if step % self._snapshot_every == 0:
                logging.info("Saving snapshot.")
                snapshot_filename = os.path.join(snapshot_dir, str(step))
                torch.save(self._model.state_dict(), snapshot_filename)
