import torch
import numpy as np

class SynWhen2StopAdapter_CF():
    def __init__(self, dataset_collection, cf_seq, y_max, y_min, precompute_a: bool = True):
        self.data_source = dataset_collection.data
        self.cf_seq = cf_seq
        self.y_max = y_max
        self.y_min = y_min
        self._precomputed_a = None
        self._common_a = None

        if precompute_a:
            self.precompute_a()

    def precompute_a(self):
        """
        Precompute the counterfactual treatment sequences `a` for each sample.
        After this runs, `__getitem__` can return `a` directly with no per-sample recomputation.
        """
        if len(self.data_source) == 0:
            self._precomputed_a = []
            self._common_a = None
            return

        # If cf_seq is not callable, it's a constant sequence shared by all samples.
        if not callable(self.cf_seq[0]):
            self._common_a = torch.tensor(self.cf_seq, dtype=torch.double).reshape(-1, 1)
            self._precomputed_a = None
            return

        precomputed = []
        for idx in range(len(self.data_source)):
            L = self.data_source[idx]["L"]
            A = self.data_source[idx]["A"]
            a = torch.zeros(len(self.cf_seq), 1, dtype=torch.double)
            for t in range(len(self.cf_seq)):
                a[t] = self.cf_seq[t](L[: t + 1], A[:t])
            precomputed.append(a)

        self._precomputed_a = precomputed
        self._common_a = None

    def __getitem__(self, idx):
        W = self.data_source[idx]['W'] # Adapting to the format of the dataset
        L = self.data_source[idx]['L']
        A = self.data_source[idx]['A'] 
        Y = np.nan * np.ones(len(A))
        Y[-1] = (self.data_source[idx]['Y'] - self.y_min) / (self.y_max - self.y_min)

        # Counterfactual treatment sequence (precomputed in preprocessing step)
        if self._common_a is not None:
            a = self._common_a
        elif self._precomputed_a is not None:
            a = self._precomputed_a[idx]
        else:
            # Fallback for backwards compatibility if precompute_a=False
            if callable(self.cf_seq[0]):
                a = torch.zeros(len(self.cf_seq), 1, dtype=torch.double)
                for t in range(len(self.cf_seq)):
                    a[t] = self.cf_seq[t](L[: t + 1], A[:t])
            else:
                a = torch.tensor(self.cf_seq, dtype=torch.double).reshape(-1, 1)

        transformed_batch = {
            'W': torch.tensor(W, dtype=torch.double).unsqueeze(0),
            'L': torch.tensor(L, dtype=torch.double).unsqueeze(1), 
            'A': torch.tensor(A, dtype=torch.double).unsqueeze(1),
            'Y': torch.tensor(Y, dtype=torch.double).unsqueeze(1),
            'a': a,
            # 'a': torch.tensor(A, dtype=torch.double).unsqueeze(1) # Just take the simulated action.
        }
        return transformed_batch

    def __len__(self):
        return len(self.data_source)

class SynWhen2StopAdapter_CF_with_policy_embedding():
    def __init__(self, dataset_collection, cf_seq, y_max, y_min, policy_embedding, precompute_a: bool = True):
        self.data_source = dataset_collection.data
        self.cf_seq = cf_seq
        self.y_max = y_max
        self.y_min = y_min
        self.policy_embedding = policy_embedding
        self._precomputed_a = None
        self._common_a = None

        if precompute_a:
            self.precompute_a()

    def precompute_a(self):
        """
        Precompute the counterfactual treatment sequences `a` for each sample.
        """
        if len(self.data_source) == 0:
            self._precomputed_a = []
            self._common_a = None
            return

        if not callable(self.cf_seq[0]):
            self._common_a = torch.tensor(self.cf_seq, dtype=torch.double).reshape(-1, 1)
            self._precomputed_a = None
            return

        precomputed = []
        for idx in range(len(self.data_source)):
            L = self.data_source[idx]["L"]
            A = self.data_source[idx]["A"]
            a = torch.zeros(len(self.cf_seq), 1, dtype=torch.double)
            for t in range(len(self.cf_seq)):
                a[t] = self.cf_seq[t](L[: t + 1], A[:t])
            precomputed.append(a)

        self._precomputed_a = precomputed
        self._common_a = None

    def __getitem__(self, idx):
        W = self.data_source[idx]['W'] # Adapting to the format of the dataset
        L = self.data_source[idx]['L']
        A = self.data_source[idx]['A'] 
        Y = np.nan * np.ones(len(A))
        Y[-1] = (self.data_source[idx]['Y'] - self.y_min) / (self.y_max - self.y_min)
        policy_embedding = self.policy_embedding

        # Counterfactual treatment sequence (precomputed in preprocessing step)
        if self._common_a is not None:
            a = self._common_a
        elif self._precomputed_a is not None:
            a = self._precomputed_a[idx]
        else:
            # Fallback if precompute_a=False
            if callable(self.cf_seq[0]):
                a = torch.zeros(len(self.cf_seq), 1, dtype=torch.double)
                for t in range(len(self.cf_seq)):
                    a[t] = self.cf_seq[t](L[: t + 1], A[:t])
            else:
                a = torch.tensor(self.cf_seq, dtype=torch.double).reshape(-1, 1)

        transformed_batch = {
            'W': torch.tensor(W, dtype=torch.double).unsqueeze(0),
            'L': torch.tensor(L, dtype=torch.double).unsqueeze(1), 
            'A': torch.tensor(A, dtype=torch.double).unsqueeze(1),
            'Y': torch.tensor(Y, dtype=torch.double).unsqueeze(1),
            'a': a,
            'policy_embedding': torch.tensor(policy_embedding, dtype=torch.double)
        }
        return transformed_batch

    def __len__(self):
        return len(self.data_source)

class MIMICDGPAdapter_CF():
    """
    Adapter for the MIMIC-Extract datasets
    """

    def __init__(self, data, cf_seq, Y_max=None, Y_min=None, precompute_a: bool = True, policy_id=None, policy_idx=None):
        self.data = torch.tensor(data, dtype=torch.float64)
        self.cf_seq = cf_seq
        self.T = self.data.shape[1]
        Y_all = self.data[:,:,0]   
        self.Y_max = Y_max if Y_max is not None else Y_all.max()
        self.Y_min = Y_min if Y_min is not None else Y_all.min()
        self.policy_id = policy_id
        self.policy_idx = policy_idx
        self._precomputed_a = None
        self._common_a = None

        if precompute_a:
            self.precompute_a()

    def __len__(self):
        return self.data.shape[0]

    def precompute_a(self):
        """
        Precompute the counterfactual treatment sequences `a` for each patient.
        """
        n = self.data.shape[0]

        if not callable(self.cf_seq[0]):
            self._common_a = torch.tensor(self.cf_seq, dtype=torch.double).reshape(-1, 1)
            self._precomputed_a = None
            return

        a_all = torch.zeros((n, len(self.cf_seq), 1), dtype=torch.double)
        for idx in range(n):
            for t in range(len(self.cf_seq)):
                a_all[idx, t] = self.cf_seq[t](
                    self.data[idx : idx + 1, : t + 1, 2:-1],
                    self.data[idx : idx + 1, :t, 0],
                    self.data[idx : idx + 1, :t, 1],
                )

        self._precomputed_a = a_all
        self._common_a = None

    def __getitem__(self, idx):
        # Extract data for single patient at index idx
        # self.data[idx] has shape (T, features) where features = [Y, A, vitals..., Y_prev]
        
        # L: dynamic covariates (vitals + Y_prev) - shape should be (T, L_dim)
        L = self.data[idx, :, 2:]  # Shape: (T, L_dim) where L_dim includes vitals and Y_prev
        
        # A: current treatments - shape should be (T, 1)
        A = self.data[idx, :, 1].reshape(-1, 1)  # Shape: (T, 1)
        
        # Y: outcomes - for mimic_extract, we observe intermediate outcomes as well
        Y = self.data[idx, :, 0].reshape(-1, 1)  # Shape: (T, 1)
        Y = (Y - self.Y_min) / (self.Y_max - self.Y_min + 1e-8)
        
        # W: static features - set to NaN for MIMIC (no static features in this format)
        W = torch.zeros((1,), dtype=torch.double)
        
        # a: counterfactual treatment sequence - shape should be (T, 1)
        if self._common_a is not None:
            a = self._common_a
        elif self._precomputed_a is not None:
            a = self._precomputed_a[idx]
        else:
            # Fallback if precompute_a=False
            if callable(self.cf_seq[0]):
                a = torch.zeros(len(self.cf_seq), 1, dtype=torch.double)
                for t in range(len(self.cf_seq)):
                    a[t] = self.cf_seq[t](
                        self.data[idx : idx + 1, : t + 1, 2:-1],
                        self.data[idx : idx + 1, :t, 0],
                        self.data[idx : idx + 1, :t, 1],
                    )
            else:
                a = torch.tensor(self.cf_seq, dtype=torch.double).reshape(-1, 1)
        
        out = {
            "W": W,
            "Y": Y,
            "A": A,
            "L": L,
            "a": a,
        }
        if self.policy_id is not None:
            out["policy_id"] = torch.tensor(self.policy_id, dtype=torch.double)
        if self.policy_idx is not None:
            out["policy_idx"] = torch.tensor(self.policy_idx, dtype=torch.long)
        return out

class MIMICDGPAdapter_CF_with_policy_embedding():
    """
    Adapter for the MIMIC-Extract datasets with policy embedding
    """

    def __init__(self, data, cf_seq, policy_embedding, Y_max=None, Y_min=None, precompute_a: bool = True, policy_id=None, policy_idx=None):
        self.data = torch.tensor(data, dtype=torch.float64)
        self.cf_seq = cf_seq
        self.policy_embedding = policy_embedding
        self.T = self.data.shape[1]
        Y_all = self.data[:,:,0]   
        self.Y_max = Y_max if Y_max is not None else Y_all.max()
        self.Y_min = Y_min if Y_min is not None else Y_all.min()
        self.policy_id = policy_id
        self.policy_idx = policy_idx
        self._precomputed_a = None
        self._common_a = None

        if precompute_a:
            self.precompute_a()

    def __len__(self):
        return self.data.shape[0]

    def precompute_a(self):
        """
        Precompute the counterfactual treatment sequences `a` for each patient.
        """
        n = self.data.shape[0]

        if not callable(self.cf_seq[0]):
            self._common_a = torch.tensor(self.cf_seq, dtype=torch.double).reshape(-1, 1)
            self._precomputed_a = None
            return

        a_all = torch.zeros((n, len(self.cf_seq), 1), dtype=torch.double)
        for idx in range(n):
            L = self.data[idx, :, 2:]  # (T, L_dim)
            for t in range(len(self.cf_seq)):
                hist_mean = torch.mean(L[t, :], dim=0)
                a_all[idx, t] = self.cf_seq[t](
                        self.data[idx : idx + 1, : t + 1, 2:-1],
                        self.data[idx : idx + 1, :t, 0],
                        self.data[idx : idx + 1, :t, 1],
                    )

        self._precomputed_a = a_all
        self._common_a = None

    def __getitem__(self, idx):
        # Extract data for single patient at index idx
        # self.data[idx] has shape (T, features) where features = [Y, A, vitals..., Y_prev]
        
        # L: dynamic covariates (vitals + Y_prev) - shape should be (T, L_dim)
        L = self.data[idx, :, 2:]  # Shape: (T, L_dim) where L_dim includes vitals and Y_prev
        
        # A: current treatments - shape should be (T, 1)
        A = self.data[idx, :, 1].reshape(-1, 1)  # Shape: (T, 1)
        
        # Y: outcomes - for mimic_extract, we observe intermediate outcomes as well
        Y = self.data[idx, :, 0].reshape(-1, 1)  # Shape: (T, 1)
        Y = (Y - self.Y_min) / (self.Y_max - self.Y_min + 1e-8)
        
        # W: static features - set to NaN for MIMIC (no static features in this format)
        W = torch.zeros((1,))
        
        # a: counterfactual treatment sequence - shape should be (T, 1)
        policy_embedding = self.policy_embedding

        if self._common_a is not None:
            a = self._common_a
        elif self._precomputed_a is not None:
            a = self._precomputed_a[idx]
        else:
            # Fallback if precompute_a=False
            if callable(self.cf_seq[0]):
                a = torch.zeros(len(self.cf_seq), 1, dtype=torch.double)
                for t in range(len(self.cf_seq)):
                    hist_mean = torch.mean(L[t, :], dim=0)
                    a[t] = self.cf_seq[t](
                        self.data[idx : idx + 1, : t + 1, 2:-1],
                        self.data[idx : idx + 1, :t, 0],
                        self.data[idx : idx + 1, :t, 1],
                    )
            else:
                a = torch.tensor(self.cf_seq, dtype=torch.double).reshape(-1, 1)

        transformed_batch = {
            'W': torch.tensor(W, dtype=torch.double),
            'L': torch.tensor(L, dtype=torch.double), 
            'A': torch.tensor(A, dtype=torch.double),
            'Y': torch.tensor(Y, dtype=torch.double),
            'a': a,
            'policy_embedding': torch.tensor(policy_embedding, dtype=torch.double)
        }
        if self.policy_id is not None:
            transformed_batch["policy_id"] = torch.tensor(self.policy_id, dtype=torch.double)
        if self.policy_idx is not None:
            transformed_batch["policy_idx"] = torch.tensor(self.policy_idx, dtype=torch.long)
        return transformed_batch