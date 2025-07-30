"""RNN model used for the MSI task.
This model is a simple RNN with a single layer of RNN cells.
The options are whether to require gradients for the input and output layers.
In the paper, input gradients are not required, but output gradients can be.
"""

from models import abstract_model
import torch
import torch.nn.functional as F


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
        theta_reg_weight=0.0,
        theta_reg_timesteps=5,  # ADDED: Number of timesteps post-onset to regularize
        theta_stimulus_mode="paral",  # ADDED: 'paral' or 'ortho'
    ):
        """Constructor.

        Args:
            hidden_size: Int. Hidden size.
            activity_decay: Float. Activity decay.
            input_feature_len: Int. Length of input features.
            output_feature_len: Int Length of output features.
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
        self._constrain_theta = theta_reg_weight > 0.0
        self._theta_reg_weight = theta_reg_weight
        self._theta_reg_timesteps = theta_reg_timesteps
        self._theta_stimulus_mode = theta_stimulus_mode

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

    def _compute_coding_direction(self, hiddens, signal):
        """Computes the coding direction via covariance."""
        # hiddens: [n_points, hidden_size]
        # signal: [n_points]
        hiddens_c = hiddens - hiddens.mean(dim=0, keepdim=True)
        signal_c = signal - signal.mean()
        signal_c = signal_c.unsqueeze(1)
        coding_direction = (hiddens_c * signal_c).mean(dim=0)
        return coding_direction

    def loss_terms(self, outputs):
        """Get dictionary of loss terms to be summed for the final loss."""
        # Mask out loss for zero labels
        mask = torch.logical_not(torch.isnan(outputs["labels"])).bool()
        mse_loss = self._loss(outputs["outputs"][mask], outputs["labels"][mask])
        loss_dict = {"mse_loss": mse_loss}
        # Theta regularization with inferred identity
        if self._constrain_theta and self._theta_reg_weight > 0:
            hiddens = outputs["hiddens"]
            all_inputs = outputs["inputs"].squeeze(2)
            _, seq_len, _ = all_inputs.shape

            # --- Infer identity and define stimuli ---
            if self._theta_stimulus_mode == "paral":
                # Identity is given by the second input feature
                identity = all_inputs[:, :, 1]
                # Both 'self' and 'other' conditions are driven by the first feature
                stimulus_for_self = all_inputs[:, :, 0]
                stimulus_for_other = all_inputs[:, :, 0]
                onset_stimulus = stimulus_for_self

            elif self._theta_stimulus_mode == "ortho":
                stimulus_for_self = all_inputs[:, :, 0]
                stimulus_for_other = all_inputs[:, :, 1]
                # Infer identity based on which stimulus channel is active
                identity = torch.zeros_like(stimulus_for_self)
                identity[stimulus_for_self != 0] = 1
                identity[stimulus_for_other != 0] = -1
                # An onset can be on either channel
                onset_stimulus = (
                    stimulus_for_self.abs() + stimulus_for_other.abs()
                )
            else:
                raise ValueError(
                    f"Unknown theta_stimulus_mode: {self._theta_stimulus_mode}"
                )

            # --- Infer Onsets ---
            stim_present = onset_stimulus != 0
            stim_present_padded = F.pad(stim_present, (1, 0), "constant", False)
            onsets = stim_present & ~stim_present_padded[:, :-1]

            onset_coords = torch.nonzero(onsets, as_tuple=False)
            if onset_coords.shape[0] == 0:
                return loss_dict  # No onsets in batch, skip regularization

            offset_losses = []
            last_valid_cos_sim = torch.tensor(0.0, device=self.device)

            for offset in range(self._theta_reg_timesteps):
                target_time_indices = onset_coords[:, 1] + offset
                batch_indices = onset_coords[:, 0]

                valid_mask = target_time_indices < seq_len
                if not valid_mask.any():
                    continue

                valid_batch_indices = batch_indices[valid_mask]
                valid_time_indices = target_time_indices[valid_mask]

                # Gather all necessary data at the valid coordinates
                h_offset = hiddens[valid_batch_indices, valid_time_indices]
                id_offset = identity[valid_batch_indices, valid_time_indices]
                s_offset_self = stimulus_for_self[
                    valid_batch_indices, valid_time_indices
                ]
                s_offset_other = stimulus_for_other[
                    valid_batch_indices, valid_time_indices
                ]

                # Create masks for 'self' and 'other' conditions
                self_mask = id_offset == 1
                other_mask = id_offset == -1

                if self_mask.sum() > 1 and other_mask.sum() > 1:
                    h_self = h_offset[self_mask]
                    s_self = s_offset_self[self_mask]

                    h_other = h_offset[other_mask]
                    s_other = s_offset_other[other_mask]

                    dir_self = self._compute_coding_direction(h_self, s_self)
                    dir_other = self._compute_coding_direction(h_other, s_other)

                    cos_sim = F.cosine_similarity(
                        dir_self, dir_other, dim=0, eps=1e-8
                    )
                    offset_losses.append(cos_sim.pow(2))
                    last_valid_cos_sim = cos_sim.detach()

            if offset_losses:
                theta_loss = torch.stack(offset_losses).mean()
                loss_dict["theta_loss"] = theta_loss
                loss_dict["loss"] = (
                    self._theta_reg_weight * theta_loss + mse_loss
                )
                loss_dict["theta_cosine"] = last_valid_cos_sim
        return loss_dict

    def scalars(self, data):
        return self.loss_terms(self.forward(data))

    @property
    def scalar_keys(self):
        keys = ("loss",)
        if self._constrain_theta:
            keys += ("theta_loss", "theta_cosine", "mse_loss")
        return keys
