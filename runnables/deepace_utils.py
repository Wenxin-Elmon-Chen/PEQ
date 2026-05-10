import numpy as np
import torch
from torch.utils.data import Dataset

class DeepACEDataAdapter_When2Stop(Dataset):
    """
    Adapter to convert SynWhen2StopAdapter_CF format to DeepACE expected format.
    
    DeepACE expects data as 3D numpy array with shape (batch_size, seq_len, input_size)
    where input_size = [Y, A, X...] (outcome, treatment, covariates)
    
    Your data format from SynWhen2StopAdapter_CF:
    {
        'W': static features (batch_size, 1, static_dim),
        'L': dynamic covariates (batch_size, seq_len, dynamic_dim), 
        'A': treatments (batch_size, seq_len, 1),
        'Y': outcomes (batch_size, seq_len, 1),
        'a': counterfactual treatment sequence (seq_len, 1)
    }
    """
    
    def __init__(self, data_source, y_max, y_min):
        """
        Args:
            data_source: data source
            y_max: maximum outcome value
            y_min: minimum outcome value
        """
        self.data_source = data_source
        self.y_max = y_max
        self.y_min = y_min

    def __len__(self):
        return len(self.data_source)
    
    def __getitem__(self, idx):
        # Get data from your adapter
        sample = self.data_source[idx]
        
        # sample is a dictionary with keys: 'W', 'L', 'A', 'Y', 'a'
        W = sample['W']  # Static features (1, static_dim)
        L = sample['L']  # Dynamic covariates (seq_len, dynamic_dim)
        A = sample['A']  # Treatments (seq_len, 1)
        Y = sample['Y']  # Outcomes (seq_len, 1)
        if self.y_min is not None and self.y_max is not None:
            Y = (Y - self.y_min) / (self.y_max - self.y_min)
        
        # Get dimensions
        seq_len = len(A)
        
        # Handle static features - ensure proper shape
        W = torch.tensor(W).unsqueeze(0)  # Make it (1, static_dim)
        L = torch.tensor(L).unsqueeze(1)  # Make it (seq_len, 1)
        
        # Repeat static features across time steps
        W_repeated = W.repeat(seq_len, 1)  # (seq_len, static_dim)

        Y_filled = torch.nan * torch.ones((seq_len, 1))
        Y_filled[-1] = torch.tensor(Y)
        
        # Concatenate in DeepACE expected order: [Y, A, W, L]
        # Shape: (seq_len, 1 + 1 + static_dim + dynamic_dim)
        deepace_format = torch.cat([
            Y_filled,           # Outcomes (seq_len, 1)
            torch.tensor(A).reshape(-1, 1),           # Treatments (seq_len, 1)  
            W_repeated,  # Static features repeated (seq_len, static_dim)
            L,            # Dynamic covariates (seq_len, dynamic_dim)
        ], dim=1)
        
        return deepace_format

class DeepACEDataAdapter_MIMICExtract(Dataset):
    """
    Adapter to convert the semi-synthetic `mimic_extract` array format into DeepACE's
    expected per-patient time-series format.

    `mimic_extract` (see `src/data/mimic_extract/simulation.py:create_dataset`) produces:
      data: (n, T, p) where p = [Y, A, vitals..., Y_prev]

    DeepACE expects, per patient, a 2D tensor of shape (T, input_size) with columns:
      [Y, A, X...]  where X are covariates (here: vitals + Y_prev)

    Important difference vs syn_when2stop:
      - `mimic_extract` has *intermediate outcomes* Y observed at every time step.
    """

    def __init__(self, raw, y_scaler=None):
        """
        Args:
            raw: numpy array of shape (n, T, p) where p = [Y, A, vitals..., Y_prev]
            y_scaler: sklearn-like scaler with .transform() / .inverse_transform() fitted on Y
        """
        self.raw = np.asarray(raw)
        self.y_scaler = y_scaler
        self._process_data_for_deepace()

    def __len__(self):
        return len(self.raw)
    
    def _process_data_for_deepace(self):
        self.data = self.raw.copy()
        # Preferred: standardize Y and rebuild Y_prev from standardized Y (DeepACE-style z-scoring).
        if self.y_scaler is not None:
            y = self.data[:, :, 0]
            y_scaled = self.y_scaler.transform(y.reshape(-1, 1)).reshape(y.shape)
            self.data[:, :, 0] = y_scaled

            # Rebuild lagged outcome from standardized Y, ignoring whatever was provided.
            y_prev = np.zeros_like(y_scaled)
            y_prev[:, 1:] = y_scaled[:, :-1]
            self.data[:, :, -1] = y_prev
            return


        
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx])

    @staticmethod
    def fit_y_scaler(*splits, y_index: int = 0):
        """
        Fit a StandardScaler on outcome Y from MIMIC splits (arrays shaped (n, T, d)).
        Uses all non-NaN Y values across all provided splits.
        """
        from sklearn.preprocessing import StandardScaler

        arrs = [np.asarray(a) for a in splits if a is not None]
        if not arrs:
            raise ValueError("No MIMIC splits provided for Y standardization.")
        if any(a.ndim != 3 for a in arrs):
            bad = next(a for a in arrs if a.ndim != 3)
            raise ValueError(f"Expected a 3D array (n, T, d) for MIMIC splits, got shape={bad.shape}")

        y = np.concatenate([a[..., y_index].reshape(-1) for a in arrs], axis=0)
        y = y[~np.isnan(y)]
        if y.size == 0:
            raise ValueError("Could not fit y_scaler: no non-NaN Y values found in provided splits.")

        scaler = StandardScaler()
        scaler.fit(y.reshape(-1, 1))
        return scaler