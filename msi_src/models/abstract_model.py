"""RNN Model."""

# pylint: disable=import-error

import abc
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.switch_backend("agg")


class AbstractModel(torch.nn.Module, metaclass=abc.ABCMeta):
    """Abstract class for rnn models."""

    def __init__(self):
        """Constructor only calls torch.nn.Module.__init__()."""
        super(AbstractModel, self).__init__()

    @abc.abstractmethod
    def loss_terms(self, outputs):
        """Get dictionary of loss terms to be summed for the final loss."""
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, data):
        """Run the model forward on data, getting outputs."""
        raise NotImplementedError

    @abc.abstractmethod
    def scalars(self):
        """Return dictionary of scalars to log."""
        raise NotImplementedError

    @abc.abstractproperty
    def scalar_keys(self):
        """Return tuple of strings, keys of self.scalars() output."""
        raise NotImplementedError

    def figures(self, data_batch):
        """Generate dict of matplotlib figures for logging."""
        rnn_outs = self.forward(data_batch)
        fig, ax = plt.subplots(2, 1, figsize=(6, 6))
        ax[0].plot(np.squeeze(rnn_outs["inputs"][0].detach().numpy()))
        ax[0].set_ylabel("Stimulus")
        ax[1].plot(np.squeeze(rnn_outs["labels"][0].detach().numpy()))
        ax[1].plot(np.squeeze(rnn_outs["outputs"][0].detach().numpy()))
        ax[1].set_ylabel("Output")
        ax[1].set_xlabel("Time")
        ax[1].legend(["truth", "model"])
        return {"input_truth_output": fig}
