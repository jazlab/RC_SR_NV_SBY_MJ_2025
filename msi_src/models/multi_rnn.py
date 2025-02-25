"""RNN model used for the MSI task.
This model is a simple RNN with a single layer of RNN cells.
The options are whether to require gradients for the input and output layers.
In the paper, input gradients are not required, but output gradients can be.
"""

from models import abstract_model
import torch


class MultiRNN(abstract_model.AbstractModel):
    """RNN model."""

    def __init__(
        self,
        hidden_size,
        activity_decay=0.1,
        input_feature_len=1,
        output_feature_len=1,
        require_input_grad=False,
        require_output_grad=False,
    ):
        """Constructor.

        Args:
            hidden_size: Int. Hidden size.
            activity_decay: Float. Activity decay.
            input_feature_len: Int. Length of input features.
            output_features_len: Int Length of output features.
        """
        super(MultiRNN, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._loss = torch.nn.MSELoss().to(self.device)

        self._hidden_size = hidden_size
        self._activity_decay = activity_decay
        self._input_feature_len = input_feature_len
        self._output_feature_len = output_feature_len

        print(
            f"Initializing MultiRNN with hidden_size: {hidden_size}, activity_decay: {activity_decay}, input_feature_len: {input_feature_len}, output_feature_len: {output_feature_len}"
        )

        self._encoder = torch.nn.Linear(
            in_features=input_feature_len,
            out_features=hidden_size,
        )

        self._encoder = self._encoder.to(self.device)

        # fix weights of encoder
        self._encoder.weight.requires_grad = require_input_grad

        self._activation = torch.nn.Tanh().to(self.device)
        self._rnn_linear = torch.nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
        ).to(self.device)
        self._decoder = torch.nn.Linear(
            in_features=hidden_size,
            out_features=output_feature_len,
            bias=True,
        ).to(self.device)

        # fix weights of decoder
        self._decoder.weight.requires_grad = require_output_grad

    def forward(self, data):
        """Run the model forward on inputs.

        Args:
            data: Dict. Must have 'inputs' item containing a batch of sequences
                of shape [batch_size, seq_len, n]. Must also have 'labels' item
                containins batch of labels of shape [batch_size, seq_len, 1].

        Returns:
            outs: Dict of outputs.
        """
        inputs = data["inputs"].to(self.device)
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)

        # Initializing hidden state
        hiddens = [self.init_hidden(batch_size)]

        # Apply RNN to get latent (hidden) states
        for i in range(seq_len):
            recent_hiddens = hiddens[-1]
            input = inputs[:, i].view(batch_size, self._input_feature_len)
            rate = self._rnn_linear(self._activation(recent_hiddens))
            embedding = self._encoder(input)
            delta_hiddens = -1 * recent_hiddens + embedding + rate
            hiddens.append(
                (1 - self._activity_decay) * recent_hiddens
                + self._activity_decay * delta_hiddens
            )

        hiddens = torch.cat([torch.unsqueeze(h, 1) for h in hiddens[1:]], dim=1)

        # Apply decoder to hiddens
        flat_hiddens = hiddens.view(batch_size * seq_len, self._hidden_size)
        outputs = self._decoder.forward(flat_hiddens)
        outputs = outputs.view(batch_size, seq_len, self._output_feature_len)

        outs = {
            "inputs": inputs,
            "outputs": outputs,
            "hiddens": hiddens,
            "labels": data["labels"].to(self.device),
        }

        return outs

    def init_hidden(self, batch_size):
        """Initialize hidden state with zeros."""
        hidden = torch.zeros(batch_size, self._hidden_size).to(self.device)
        return hidden

    def loss_terms(self, outputs):
        """Get dictionary of loss terms to be summed for the final loss."""
        # Mask out loss for zero labels
        mask = torch.logical_not(torch.isnan(outputs["labels"])).bool()
        loss = self._loss(outputs["outputs"][mask], outputs["labels"][mask])
        return {"loss": loss}

    def scalars(self, data):
        return self.loss_terms(self.forward(data))

    @property
    def scalar_keys(self):
        return ("loss",)
