"""Multi-Source-Integration (MSI) task.

To demo this task, run demo_msi_task.py.
"""

import numpy as np


class MSI(object):
    """Multi-Source-Integration task.

    This task has a stimulus consisting of two inputs representing evidence from
    two sources. The first source is self, and the second source is other.

    The task is to integrate the two sources of evidence and accumulate over
    trials.
    """

    def __init__(
        self,
        timesteps,
        model_type,
        integration_factor,
        mean_choices,
        value_std,
        test=False,
        iti=5,
        rng=None,
    ):
        """Constructor.

        Args:
            timesteps: Int. Number of timesteps in the total trial.
            model_type: String. One of 'ortho', 'paral.
        """
        self._timesteps = timesteps
        self._model_type = model_type
        self._integration_factor = integration_factor
        self._mean_choices = mean_choices
        self._value_std = value_std
        self._test = test
        self._risetime = iti
        if rng is None:
            rng = np.random.default_rng(12345)  # for reproducing results
        else:
            rng = rng

    def generate_identity(self, trials):
        # returns a sequence of identity for each time point
        # for input timepoints, 1 for self, -1 for other
        # for other time points, 0
        tstarts = trials["trialstarts"]
        ntrials = len(tstarts)
        len_sequence = self._timesteps
        identities = np.ones(ntrials)
        if self._test:
            identities[ntrials // 2 :] = -1
        else:
            identities = np.random.choice([1, -1], size=ntrials)
        sequence = np.zeros(len_sequence)
        for ts, id in zip(tstarts, identities):
            sequence[ts] = id
        return sequence

    def generate_means(self, trials):
        tstarts = trials["trialstarts"]
        ntrials = len(tstarts)
        len_sequence = self._timesteps
        means = np.random.choice(self._mean_choices, size=ntrials)
        if self._test:
            means[: ntrials // 4] = self._mean_choices[0]
            means[ntrials // 4 : ntrials // 2] = self._mean_choices[1]
            means[ntrials // 2 : 3 * ntrials // 4] = self._mean_choices[0]
            means[3 * ntrials // 4 :] = self._mean_choices[1]
        sequence = np.zeros(len_sequence)
        for ts, m in zip(tstarts, means):
            sequence[ts] = m
        return sequence

    def generate_trials(self):
        n = self._timesteps
        trials = {"trialstarts": [], "rampstarts": []}
        start = 0
        ramp = self._risetime
        max_len = ramp * 4
        min_len = ramp * 2
        while start + max_len < n:
            t_trial = np.random.randint(min_len, max_len)
            # randomize both delay and iti with constraint delay+iti=t_trial-2*ramp
            total_iti = t_trial - 2 * ramp
            delay = np.random.randint(0, total_iti) if total_iti > 0 else 0
            trials["trialstarts"].append(start)
            trials["rampstarts"].append(start + delay)
            start += t_trial
        return trials

    def __call__(self):
        """Return a stimulus and desired response.

        Returns:
            Dictionary with keys 'inputs' and 'labels'. 'inputs' contains a
                time-like stimulus. 'labels' contains a time-like response.
                A model that solves this task should take in 'inputs' and
                produce 'labels'.
        """

        # Generate mean and identity
        trials = self.generate_trials()
        identity = self.generate_identity(trials)
        means = self.generate_means(trials)
        values = np.random.normal(means, self._value_std)
        # set value to zero where identity is zero
        values[identity == 0] = 0
        # Generate stimulus
        if self._model_type == "ortho":
            stim_self = [
                values[i] if iden == 1 else 0 for i, iden in enumerate(identity)
            ]
            stim_other = [
                values[i] if iden == -1 else 0
                for i, iden in enumerate(identity)
            ]
            inputs = np.column_stack([stim_self, stim_other])
        elif self._model_type == "paral":
            stim_value = values
            stim_identity = identity
            inputs = np.column_stack([stim_value, stim_identity])

        # determine integration factor for each time point
        integration_factors = np.zeros(self._timesteps)
        integration_factors[identity == 1] = 1
        integration_factors[identity == -1] = self._integration_factor
        scaled_input = values * integration_factors

        # Generate accumulated values
        accumulator = 0
        output = []
        for i, value in enumerate(values):
            if value > 0:
                accumulator = 0
            else:
                accumulator += scaled_input[i]
            output.append(accumulator)
        output = np.array(output)

        # mask accumulated output with a report ramp:
        input_report = np.zeros(self._timesteps)
        for ramp_start in trials["rampstarts"]:
            ramp_end = ramp_start + self._risetime
            report_off = ramp_end + self._risetime
            input_report[ramp_start:ramp_end] = np.linspace(
                0, 1, self._risetime
            )
            input_report[ramp_end:report_off] = 1
        # output set to np.nan when input_report is not 1
        output = [
            o if report == 1 else np.nan
            for o, report in zip(output, input_report)
        ]

        # add go signal to inputs
        inputs = np.column_stack([inputs, input_report])
        inputs = np.expand_dims(inputs, 1).astype(np.float32)
        labels = np.expand_dims(output, 1).astype(np.float32)
        return {
            "inputs": inputs,
            "labels": labels,
            "identity": identity,
            "trialstarts": np.array(trials["trialstarts"]),
            "rampstarts": np.array(trials["rampstarts"]),
        }

    @property
    def data_keys(self):
        return ("inputs", "labels")
