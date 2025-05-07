import scipy
import numpy as np
from sklearn.metrics import r2_score


class BaseFemaleFly:
    def __init__(self):
        self.fit_success = None

    def fit(self, emissions, inputs, output_mn_std=None):
        raise NotImplementedError

    def update_status(self):
        self.fit_success = self.check_nan_in_fit_params()
        print("SUCCEEDED?", self.fit_success)
        return

    def check_nan_in_fit_params(self):
        raise NotImplementedError

    def predict(self, emissions, inputs):
        raise NotImplementedError

    def score(self, emissions, inputs):
        y_preds, _ = self.predict(emissions, inputs)
        y_preds = np.concatenate(y_preds, axis=0)
        emissions = np.concatenate(emissions, axis=0)
        return r2_score(emissions, y_preds)

    def score_by_o(self, emissions, inputs):
        y_preds, _ = self.predict(emissions, inputs)
        y_preds = np.concatenate(y_preds, axis=0)
        emissions = np.concatenate(emissions, axis=0)
        return {o: r2_score(emissions[:, o], y_preds[:, o]) for o in range(self.data_config["emission_dim"])}

    def score_by_z(self, emissions, inputs):
        y_preds, z_seqs = self.predict(emissions, inputs)
        y_preds = np.concatenate(y_preds, axis=0)
        emissions = np.concatenate(emissions, axis=0)
        z_seqs = np.concatenate(z_seqs, axis=0)
        # print("z_seqs unique", np.unique(z_seqs, return_counts=True))
        return {z: (r2_score(emissions[z_seqs == z], y_preds[z_seqs == z]) if np.any(z_seqs == z) else 0.0) for z in range(self.num_states)}

    def score_by_z_and_o(self, emissions, inputs):
        y_preds, z_seqs = self.predict(emissions, inputs)
        y_preds = np.concatenate(y_preds, axis=0)
        emissions = np.concatenate(emissions, axis=0)
        z_seqs = np.concatenate(z_seqs, axis=0)
        return {z: {o: (r2_score(emissions[z_seqs == z][:, o], y_preds[z_seqs == z][:, o]) if np.any(z_seqs == z) else 0.0) for o in range(self.data_config["emission_dim"])}
                for z in range(self.num_states)}

    def correlation_by_o(self, emissions, inputs):
        y_preds, _ = self.predict(emissions, inputs)
        y_preds = np.concatenate(y_preds, axis=0)
        emissions = np.concatenate(emissions, axis=0)
        corrs_dict = {}
        for o in range(self.data_config["emission_dim"]):
            a = y_preds[:, o]
            b = emissions[:, o]
            c = scipy.signal.correlate(a, b, mode='valid') / (np.linalg.norm(a) * np.linalg.norm(b))
            corrs_dict[o] = c[0]     # zero lag correlation only at the moment
        return corrs_dict
