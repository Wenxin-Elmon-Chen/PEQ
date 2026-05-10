import numpy as np
import pandas as pd
import os
from scipy.special import expit

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def standardize_covariates(x):
    """
    DeepACE-style standardization:
    Standardize covariates per variable across all samples & timesteps.
    x: (n, T, p)
    """
    mean = np.nanmean(x, axis=(0, 1), keepdims=True)
    std = np.nanstd(x, axis=(0, 1), keepdims=True) + 1e-6
    return (x - mean) / std


def simulate_treat_outcomes(config, a, data_vitals, seed=42):
    np.random.seed(seed)
    n = config["n"]
    T = config["T"]
    h = config["lag"]

    A_f = np.zeros((n, T))
    a_cf = np.tile(np.expand_dims(a, 0), (n, 1))
    Y_f = np.zeros((n, T))
    Y_cf = np.zeros((n, T))

    # print(np.mean(data_vitals))

    # Noise
    def noise(s, sd):
        return np.random.normal(loc=0, scale=sd, size=s)

    err_A = noise((n, T), config["noise_A"])
    err_Y = noise((n, T), config["noise_Y"])

    # Coefficients
    coef_xa = np.empty(h)
    coef_ya = np.empty(h)
    coef_xy = np.empty(h)
    coef_ay = np.empty(h)
    coef_yy = np.empty(h)
    for i in range(h):
        coef_xa[i] = ((-1) ** i) * (1 / (i + 1))
        coef_ya[i] = ((-1) ** i) * (1 / (i + 1))
        coef_xy[i] = ((-1) ** i) * (1 / (i + 1))
        coef_ay[i] = ((-1) ** i) * (1 / (i + 1))
        coef_yy[i] = ((-1) ** i) * (1 / (i + 1))

    # Data generation
    def generate_data(A=None, cf=False):
        if A is None:
            A = np.zeros((n, T))
        Y = np.zeros((n, T))
        treat_level_mid = T / 2
        treat_level_max = T
        treat_level = np.full(n, treat_level_mid - 3)

        for t in range(0, T):
            t_start = max(0, t - h)
            hist = data_vitals[:, t_start:(t + 1), :]
            hist_y = np.tanh(Y[:, t_start:t] / 2)
            hist_mean = np.mean(hist, axis=2)

            if not cf:
                # Treatment assignment
                a_contrib = err_A[:, t] - np.tanh(treat_level - treat_level_mid)
                for i in range(min(h, t + 1)):
                    a_contrib += coef_xa[i] * hist_mean[:, hist_mean.shape[1] - 1 - i]
                for i in range(min(h - 1, t)):
                    a_contrib += coef_ya[i] * hist_y[:, hist_y.shape[1] - 1 - i]
                prob_A = expit(a_contrib)
                A[:, t] = np.where(prob_A > 0.5, 1, 0)

            # Adjust patient medication level
            if t > 1:
                treat_level += (2 * A[:, t] - np.ones(n)) * np.abs(
                    np.mean(data_vitals[:, t, :], axis=1) * np.tanh(Y[:, t - 1])
                )
            else:
                treat_level += (2 * A[:, t] - np.ones(n)) * np.abs(
                    np.mean(data_vitals[:, t, :], axis=1)
                )
            if t > 0:
                treat_level += (2 * A[:, t - 1] - np.ones(n))

            for i in range(n):
                if treat_level[i] < 0:
                    treat_level[i] = 0
                if treat_level[i] > treat_level_max:
                    treat_level[i] = treat_level_max

            # Outcomes
            base_term = err_Y[:, t]
            Y[:, t] = base_term
            past_term = 0
            for i in range(h):
                if t - i >= 0:
                    past_term += coef_ya[i] * np.tanh(
                        np.sin(np.mean(data_vitals[:, t - i, 0:5], axis=1)) * A[:, t - i]
                        + np.cos(np.mean(data_vitals[:, t - i, 5:], axis=1)) * A[:, t - i]
                    )
            Y[:, t] = 5 * past_term + base_term

        return A, Y

    A_f, Y_f = generate_data(cf=False)
    A_cf, Y_cf = generate_data(a_cf, cf=True)

    def create_dataset(Y, A):
        data = np.concatenate((np.expand_dims(Y, 2), np.expand_dims(A, 2), data_vitals), axis=2)
        Y_prev = Y.copy()
        Y_prev[:, 0] = np.zeros(n)
        Y_prev[:, 1:] = Y[:, 0:(T - 1)]
        data = np.concatenate((data, np.expand_dims(Y_prev, 2)), 2)
        return data

    return create_dataset(Y_f, A_f), create_dataset(Y_cf, A_cf)


def simulate_treatment_outcomes_function(config, cf_seq, data_vitals, seed=42):
    """
    Same as `simulate_treat_outcomes`, except the *counterfactual* treatment A_cf is assigned
    by a user-provided policy function instead of a fixed treatment sequence.

    The policy signature is:
        treatment_policy(hist, hist_y) -> {0,1} or probability in [0,1]

    where
      - hist:   np.ndarray of shape (n, window_len, p)   (vitals history up to current t)
      - hist_y: np.ndarray of shape (n, window_len-1)    (outcome history up to t-1, transformed)

    It may return:
      - scalar (applied to all n), or
      - array-like of shape (n,) or (n,1)
    """
    if not callable(cf_seq[0]):
        raise ValueError("treatment_policy must be callable: f(hist, hist_y) -> 0/1 (or prob)")

    np.random.seed(seed)
    n = config["n"]
    T = config["T"]
    h = config["lag"]

    A_f = np.zeros((n, T))
    Y_f = np.zeros((n, T))
    A_cf = np.zeros((n, T))
    Y_cf = np.zeros((n, T))

    # Noise (same as simulate_treat_outcomes)
    def noise(s, sd):
        return np.random.normal(loc=0, scale=sd, size=s)

    err_A = noise((n, T), config["noise_A"])
    err_Y = noise((n, T), config["noise_Y"])

    # Coefficients (same as simulate_treat_outcomes)
    coef_xa = np.empty(h)
    coef_ya = np.empty(h)
    coef_xy = np.empty(h)
    coef_ay = np.empty(h)
    coef_yy = np.empty(h)
    for i in range(h):
        coef_xa[i] = ((-1) ** i) * (1 / (i + 1))
        coef_ya[i] = ((-1) ** i) * (1 / (i + 1))
        coef_xy[i] = ((-1) ** i) * (1 / (i + 1))
        coef_ay[i] = ((-1) ** i) * (1 / (i + 1))
        coef_yy[i] = ((-1) ** i) * (1 / (i + 1))

    # def _policy_to_binary(policy_out):
    #     """Convert policy output to binary vector shape (n,)."""
    #     out = np.asarray(policy_out)

    #     # scalar -> broadcast
    #     if out.ndim == 0:
    #         out = np.full((n,), out.item())
    #     # (n,1) -> (n,)
    #     if out.ndim == 2 and out.shape == (n, 1):
    #         out = out[:, 0]
    #     if out.shape != (n,):
    #         raise ValueError(
    #             f"treatment_policy must return scalar or shape (n,) / (n,1). Got shape {out.shape}."
    #         )

    #     if out.dtype == bool:
    #         return out.astype(float)

    #     # probability-like -> threshold
    #     if np.issubdtype(out.dtype, np.floating):
    #         return (out > 0.5).astype(float)

    #     # integer-like -> clip to {0,1}
    #     return np.clip(out.astype(float), 0.0, 1.0)

    # Data generation (same structure as simulate_treat_outcomes)
    def generate_data_factual():
        A = np.zeros((n, T))
        Y = np.zeros((n, T))
        treat_level_mid = T / 2
        treat_level_max = T
        treat_level = np.full(n, treat_level_mid - 3)

        for t in range(0, T):
            t_start = max(0, t - h)
            hist = data_vitals[:, t_start:(t + 1), :]
            hist_y = np.tanh(Y[:, t_start:t] / 2)
            hist_mean = np.mean(hist, axis=2)

            # Treatment assignment (factual only)
            a_contrib = err_A[:, t] - np.tanh(treat_level - treat_level_mid)
            for i in range(min(h, t + 1)):
                a_contrib += coef_xa[i] * hist_mean[:, hist_mean.shape[1] - 1 - i]
            for i in range(min(h - 1, t)):
                a_contrib += coef_ya[i] * hist_y[:, hist_y.shape[1] - 1 - i]
            prob_A = expit(a_contrib)
            A[:, t] = np.where(prob_A > 0.5, 1.0, 0.0)

            # Adjust patient medication level
            if t > 1:
                treat_level += (2 * A[:, t] - np.ones(n)) * np.abs(
                    np.mean(data_vitals[:, t, :], axis=1) * np.tanh(Y[:, t - 1])
                )
            else:
                treat_level += (2 * A[:, t] - np.ones(n)) * np.abs(
                    np.mean(data_vitals[:, t, :], axis=1)
                )
            if t > 0:
                treat_level += (2 * A[:, t - 1] - np.ones(n))

            for i in range(n):
                if treat_level[i] < 0:
                    treat_level[i] = 0
                if treat_level[i] > treat_level_max:
                    treat_level[i] = treat_level_max

            # Outcomes
            base_term = err_Y[:, t]
            past_term = 0
            for i in range(h):
                if t - i >= 0:
                    past_term += coef_ya[i] * np.tanh(
                        np.sin(np.mean(data_vitals[:, t - i, 0:5], axis=1)) * A[:, t - i]
                        + np.cos(np.mean(data_vitals[:, t - i, 5:], axis=1)) * A[:, t - i]
                    )
            Y[:, t] = 5 * past_term + base_term

        return A, Y

    def generate_data_counterfactual_with_policy():
        A = np.zeros((n, T))
        Y = np.zeros((n, T))
        treat_level_mid = T / 2
        treat_level_max = T
        treat_level = np.full(n, treat_level_mid - 3)

        for t in range(0, T):
            # t_start = max(0, t - h)
            # hist = data_vitals[:, t_start:(t + 1), :]
            # hist_y = np.tanh(Y[:, t_start:t] / 2)
            
            # policy_out = cf_seq[t](hist, hist_y, treat_level)
            policy_out = cf_seq[t](data_vitals[:, : (t + 1), :], Y[:, :t], A[:, :t])

            A[:, t] = policy_out

            # Adjust patient medication level (same as simulate_treat_outcomes)
            if t > 1:
                treat_level += (2 * A[:, t] - np.ones(n)) * np.abs(
                    np.mean(data_vitals[:, t, :], axis=1) * np.tanh(Y[:, t - 1])
                )
            else:
                treat_level += (2 * A[:, t] - np.ones(n)) * np.abs(
                    np.mean(data_vitals[:, t, :], axis=1)
                )
            if t > 0:
                treat_level += (2 * A[:, t - 1] - np.ones(n))

            for i in range(n):
                if treat_level[i] < 0:
                    treat_level[i] = 0
                if treat_level[i] > treat_level_max:
                    treat_level[i] = treat_level_max

            # Outcomes (same as simulate_treat_outcomes)
            base_term = err_Y[:, t]
            past_term = 0
            for i in range(h):
                if t - i >= 0:
                    past_term += coef_ya[i] * np.tanh(
                        np.sin(np.mean(data_vitals[:, t - i, 0:5], axis=1)) * A[:, t - i]
                        + np.cos(np.mean(data_vitals[:, t - i, 5:], axis=1)) * A[:, t - i]
                    )
            Y[:, t] = 5 * past_term + base_term

        return A, Y

    A_f, Y_f = generate_data_factual()
    A_cf, Y_cf = generate_data_counterfactual_with_policy()

    def create_dataset(Y, A):
        data = np.concatenate((np.expand_dims(Y, 2), np.expand_dims(A, 2), data_vitals), axis=2)
        Y_prev = Y.copy()
        Y_prev[:, 0] = np.zeros(n)
        Y_prev[:, 1:] = Y[:, 0:(T - 1)]
        data = np.concatenate((data, np.expand_dims(Y_prev, 2)), 2)
        return data

    return create_dataset(Y_f, A_f), create_dataset(Y_cf, A_cf)




def load_mimic_covariates(n, T, vital_list, static_list, mimic_extract_file, seed):
    
    h5 = pd.HDFStore(mimic_extract_file, 'r')

    all_vitals = h5['/vitals_labs_mean'][vital_list].copy()

    all_vitals = all_vitals.droplevel(['hadm_id', 'icustay_id'])

    column_names = []
    for column in all_vitals.columns:
        if isinstance(column, str):
            column_names.append(column)
        else:
            column_names.append(column[0])
    all_vitals.columns = column_names

    # Filling NA 
    all_vitals = all_vitals.fillna(method='ffill')
    all_vitals = all_vitals.fillna(method='bfill')

    # Static features
    static_features = None
    if static_list is not None:
        static_features = h5[static_list].copy()
        static_features = static_features.droplevel(['hadm_id', 'icustay_id'])

    # Filtering out users with time length < 2T 
    user_sizes = all_vitals.groupby('subject_id').size()
    filtered_users_len = user_sizes.index[user_sizes >= 2 * T]

    if static_list is not None:
        if "age" in static_list:
            filtered_users_age = static_features.index[static_features.age < 100]
            filtered_users = filtered_users_len.intersection(filtered_users_age)
        else:
            filtered_users = filtered_users_len
    else:
        filtered_users = filtered_users_len

    np.random.seed(seed)
    filtered_users = np.random.choice(filtered_users, size=n, replace=False)
    all_vitals = all_vitals.loc[filtered_users]

    vitals_grouped = all_vitals.groupby('subject_id')
    data_vitals = np.zeros((n, T, len(vital_list)))

    for i, cov in enumerate(vitals_grouped):
        test = cov[1].to_numpy()
        for t in range(T):
            data_vitals[i, t, :] = test[t, :]

    # Standardize
    data_vitals = standardize_covariates(data_vitals)

    # Process static features
    if static_list is not None:
        static_features = static_features.loc[filtered_users]
        processed_static_features = []
        for feature in static_features.columns:
            if not isinstance(static_features[feature].iloc[0], float):
                one_hot = pd.get_dummies(static_features[feature])
                processed_static_features.append(one_hot.astype(float))
            else:
                mean = np.mean(static_features[feature])
                std = np.std(static_features[feature])
                processed_static_features.append((static_features[feature] - mean) / std)
        static_features = pd.concat(processed_static_features, axis=1).to_numpy()
    else:
        static_features = None

    return data_vitals, static_features


if __name__ == "__main__":

    
    n = 1000
    T = 48

    vital_list = [
        ('Heart Rate', 'mean'),
        ('Sodium', 'mean'),
        ('Mean blood pressure', 'mean'),
        ('Glucose', 'mean'),
        ('Hematocrit', 'mean'),
        ('Respiratory rate', 'mean'),
        ('Prothrombin time PT', 'mean'),
        ('Hemoglobin', 'mean'),
        ('Creatinine', 'mean'),
        ('Blood urea nitrogen', 'mean'),
    ]

    static_list = None   

    print("Loading vitals from MultiIndex dataframe ...")
    data_vitals, _ = load_mimic_covariates(n, T, vital_list, static_list)

    print("data_vitals loaded. Shape =", data_vitals.shape)
    
    config = {
        "n": n,
        "T": T,
        "lag": 5,
        "noise_A": 0.1,
        "noise_Y": 0.1,
    }

    a_int_1 = np.zeros(T)
    a_int_2 = np.ones(T)

    print("Simulating DeepACE factual & counterfactual data ...")

    factual, cf1 = simulate_treat_outcomes(config, a_int_1, data_vitals)
    _, cf2 = simulate_treat_outcomes(config, a_int_2, data_vitals)

   
    #np.save(os.path.join(THIS_DIR, "factual.npy"), factual)
    #np.save(os.path.join(THIS_DIR, "cf1.npy"), cf1)
    #np.save(os.path.join(THIS_DIR, "cf2.npy"), cf2)


