import numpy as np
import pandas as pd
import os
from scipy.special import expit

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def standardize_covariates(x: np.ndarray) -> np.ndarray:
    """
    DeepACE-style standardization:
    Standardize covariates per variable across all samples & timesteps.
    x: (n, T, p)
    """
    mean = np.nanmean(x, axis=(0, 1), keepdims=True)
    std = np.nanstd(x, axis=(0, 1), keepdims=True) + 1e-6
    return (x - mean) / std


def simulate_treat_outcomes(config, a, base_vitals, seed: int = 42):
    """
    Complex semi-synthetic MIMIC-Extract DGP.

    Input:
      base_vitals: (n, T, 10) standardized MIMIC vitals (unaffected by actions/outcomes).

    Output array layout per patient/time (kept compatible with adapters):
      raw[t] = [Y_t, A_t, vitals_t..., Y_prev_t]
    where vitals_t now contains:
      [10 MIMIC vitals, 5 simulated action-affected variables]  => 15 time-varying variables total.
    """
    np.random.seed(seed)
    n = int(config["n"])
    T = int(config["T"])
    h = int(config["lag"])

    # Counterfactual fixed treatment sequence broadcast to (n, T)
    a_cf = np.tile(np.expand_dims(np.asarray(a, dtype=float), 0), (n, 1))

    # Noise
    def noise(s, sd):
        return np.random.normal(loc=0.0, scale=sd, size=s)

    err_A = noise((n, T), float(config["noise_A"]))
    err_Y = noise((n, T), float(config["noise_Y"]))

    # Coefficients (same pattern as mimic_extract)
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

    def _update_z(z_t: np.ndarray, a_t: np.ndarray, v_t: np.ndarray, y_t: np.ndarray, err: np.ndarray) -> np.ndarray:
        """
        z_t: (n, num_z)
        a_t: (n,)
        v_t: (n, dim_vitals)
        y_t: (n,)
        err: (n, num_z)
        """
        v_proj = np.mean(v_t, axis=1)  # (n,)
        return (
            config["z_ar"] * z_t
            + config["z_a_coef"] * a_t[:, None] * expit(z_t**2)[None, :]
            + config["z_v_coef"] * v_proj[:, None]
            + err
        )

    def generate_data(A=None, cf=False):
        # A_fixed: (n, T) if provided (for fixed counterfactual sequence)
        if A is None:
            A = np.zeros((n, T), dtype=float)
        Y = np.zeros((n, T), dtype=float)
        treat_level_mid = T / 2
        treat_level_max = T
        treat_level = np.full(n, treat_level_mid - 3.0, dtype=float)

        # Action-affected covariates Z: (n, T, num_z)
        Z = np.zeros((n, T, config["num_z"]), dtype=float)
        Z[:, 0, :] = np.random.normal(loc=0.0, scale=1.0, size=(n, config["num_z"]))

        for t in range(0, T):
            t_start = max(0, t - h)

            # Current full covariates (base + action-affected)
            full_cov_hist = np.concatenate(
                [base_vitals[:, t_start : (t + 1), :], Z[:, t_start : (t + 1), :]],
                axis=2,
            )
            hist_y = np.tanh(Y[:, t_start:t] / 2.0)
            hist_mean = np.mean(full_cov_hist, axis=2)

            if not cf:
                # Treatment assignment
                a_contrib = err_A[:, t] - np.tanh(treat_level - treat_level_mid)
                for i in range(min(h, t + 1)):
                    a_contrib += coef_xa[i] * hist_mean[:, hist_mean.shape[1] - 1 - i]
                for i in range(min(h - 1, t)):
                    a_contrib += coef_ya[i] * hist_y[:, hist_y.shape[1] - 1 - i]
                prob_A = expit(a_contrib)
                A[:, t] = np.where(prob_A > 0.5, 1.0, 0.0)

            # Adjust patient medication level (use mean of full covariates at time t)
            full_cov_t = np.concatenate([base_vitals[:, t, :], Z[:, t, :]], axis=1)  # (n, 10+num_z)
            if t > 1:
                treat_level += (2.0 * A[:, t] - 1.0) * np.abs(
                    np.mean(full_cov_t, axis=1) * np.tanh(Y[:, t - 1])
                )
            else:
                treat_level += (2.0 * A[:, t] - 1.0) * np.abs(np.mean(full_cov_t, axis=1))
            if t > 0:
                treat_level += (2.0 * A[:, t - 1] - 1.0)

            treat_level = np.clip(treat_level, 0.0, float(treat_level_max))

            # Outcomes (extends mimic_extract: depends on base vitals + action-affected covariates)
            base_term = err_Y[:, t]
            past_term = 0.0
            for i in range(h):
                if t - i >= 0:
                    # Keep the original split, but now "rest" includes both base vitals[5:] and Z.
                    base_part = np.mean(base_vitals[:, t - i, 0:5], axis=1)
                    rest_part = np.mean(np.concatenate([base_vitals[:, t - i, 5:], Z[:, t - i, :]], axis=1), axis=1)
                    past_term += coef_ya[i] * np.tanh(np.sin(base_part) * A[:, t - i] + np.cos(rest_part) * A[:, t - i])

            Y[:, t] = 5.0 * past_term + base_term

            # Update Z_{t+1} after generating Y_t (so Z depends on A_t and Y_t)
            if t < T - 1:
                err_Z = np.random.normal(loc=0.0, scale=config["noise_Z"], size=(n, config["num_z"]))
                Z[:, t + 1, :] = _update_z(Z[:, t, :], A[:, t], base_vitals[:, t, :], Y[:, t], err_Z)

        full_vitals = np.concatenate([base_vitals, Z], axis=2)  # (n, T, 15)

        return A, Y, full_vitals

    A_f, Y_f, full_vitals_f = generate_data(cf=False)
    A_cf, Y_cf, full_vitals_cf = generate_data(A=a_cf, cf=True)

    def create_dataset(Y, A, full_vitals):
        data = np.concatenate((np.expand_dims(Y, 2), np.expand_dims(A, 2), full_vitals), axis=2)
        Y_prev = Y.copy()
        Y_prev[:, 0] = np.zeros(n)
        Y_prev[:, 1:] = Y[:, 0 : (T - 1)]
        data = np.concatenate((data, np.expand_dims(Y_prev, 2)), 2)
        return data

    return create_dataset(Y_f, A_f, full_vitals_f), create_dataset(Y_cf, A_cf, full_vitals_cf)


def simulate_treatment_outcomes_function(config: dict, cf_seq, base_vitals: np.ndarray, seed: int = 42):
    """
    Same as `simulate_treat_outcomes`, except the *counterfactual* treatment A_cf is assigned
    by a user-provided policy function instead of a fixed treatment sequence.

    The policy signature is (kept compatible with existing code):
        treatment_policy(cov_hist, y_hist, a_hist) -> {0,1} or probability in [0,1]
    where cov_hist is (n, t+1, p_cov) and p_cov = 15 here (10 base + 5 action-affected vars).
    """
    if not callable(cf_seq[0]):
        raise ValueError("treatment_policy must be callable: f(cov_hist, y_hist, a_hist) -> 0/1 (or prob)")

    np.random.seed(seed)
    n = int(config["n"])
    T = int(config["T"])
    h = int(config["lag"])

    def noise(s, sd):
        return np.random.normal(loc=0.0, scale=sd, size=s)

    err_A = noise((n, T), float(config["noise_A"]))
    err_Y = noise((n, T), float(config["noise_Y"]))

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

    def _update_z(z_t: np.ndarray, a_t: np.ndarray, v_t: np.ndarray, y_t: np.ndarray, err: np.ndarray) -> np.ndarray:
        """
        z_t: (n, num_z)
        a_t: (n,)
        v_t: (n, dim_vitals)
        y_t: (n,)
        err: (n, num_z)
        """
        v_proj = np.mean(v_t, axis=1)  # (n,)
        return (
            config["z_ar"] * z_t
            + config["z_a_coef"] * a_t[:, None] * expit(z_t**2)[None, :]
            + config["z_v_coef"] * v_proj[:, None]
            + err
        )

    def generate_data_factual():
        A = np.zeros((n, T), dtype=float)
        Y = np.zeros((n, T), dtype=float)
        Z = np.zeros((n, T, config["num_z"]), dtype=float)
        Z[:, 0, :] = np.random.normal(loc=0.0, scale=1.0, size=(n, config["num_z"]))

        treat_level_mid = T / 2
        treat_level_max = T
        treat_level = np.full(n, treat_level_mid - 3.0, dtype=float)

        for t in range(0, T):
            t_start = max(0, t - h)
            full_cov_hist = np.concatenate(
                [base_vitals[:, t_start : (t + 1), :], Z[:, t_start : (t + 1), :]],
                axis=2,
            )
            hist_y = np.tanh(Y[:, t_start:t] / 2.0)
            hist_mean = np.mean(full_cov_hist, axis=2)

            a_contrib = err_A[:, t] - np.tanh(treat_level - treat_level_mid)
            for i in range(min(h, t + 1)):
                a_contrib += coef_xa[i] * hist_mean[:, hist_mean.shape[1] - 1 - i]
            for i in range(min(h - 1, t)):
                a_contrib += coef_ya[i] * hist_y[:, hist_y.shape[1] - 1 - i]
            prob_A = expit(a_contrib)
            A[:, t] = np.where(prob_A > 0.5, 1.0, 0.0)

            full_cov_t = np.concatenate([base_vitals[:, t, :], Z[:, t, :]], axis=1)
            if t > 1:
                treat_level += (2.0 * A[:, t] - 1.0) * np.abs(
                    np.mean(full_cov_t, axis=1) * np.tanh(Y[:, t - 1])
                )
            else:
                treat_level += (2.0 * A[:, t] - 1.0) * np.abs(np.mean(full_cov_t, axis=1))
            if t > 0:
                treat_level += (2.0 * A[:, t - 1] - 1.0)
            treat_level = np.clip(treat_level, 0.0, float(treat_level_max))

            base_term = err_Y[:, t]
            past_term = 0.0
            for i in range(h):
                if t - i >= 0:
                    base_part = np.mean(base_vitals[:, t - i, 0:5], axis=1)
                    rest_part = np.mean(np.concatenate([base_vitals[:, t - i, 5:], Z[:, t - i, :]], axis=1), axis=1)
                    past_term += coef_ya[i] * np.tanh(np.sin(base_part) * A[:, t - i] + np.cos(rest_part) * A[:, t - i])

            Y[:, t] = 5.0 * past_term + base_term

            if t < T - 1:
                err_Z = np.random.normal(loc=0.0, scale=config["noise_Z"], size=(n, config["num_z"]))
                Z[:, t + 1, :] = _update_z(Z[:, t, :], A[:, t], base_vitals[:, t, :], Y[:, t], err_Z)

        full_vitals = np.concatenate([base_vitals, Z], axis=2)
        return A, Y, full_vitals

    def generate_data_counterfactual_with_policy():
        A = np.zeros((n, T), dtype=float)
        Y = np.zeros((n, T), dtype=float)
        Z = np.zeros((n, T, config["num_z"]), dtype=float)
        Z[:, 0, :] = np.random.normal(loc=0.0, scale=1.0, size=(n, config["num_z"]))

        treat_level_mid = T / 2
        treat_level_max = T
        treat_level = np.full(n, treat_level_mid - 3.0, dtype=float)

        for t in range(0, T):
            # Build covariate history available to the policy at time t (includes action-affected vars so far)
            cov_hist = np.concatenate([base_vitals[:, : (t + 1), :], Z[:, : (t + 1), :]], axis=2)
            policy_out = cf_seq[t](cov_hist, Y[:, :t], A[:, :t])
            A[:, t] = np.asarray(policy_out, dtype=float).reshape(-1)

            full_cov_t = np.concatenate([base_vitals[:, t, :], Z[:, t, :]], axis=1)
            if t > 1:
                treat_level += (2.0 * A[:, t] - 1.0) * np.abs(
                    np.mean(full_cov_t, axis=1) * np.tanh(Y[:, t - 1])
                )
            else:
                treat_level += (2.0 * A[:, t] - 1.0) * np.abs(np.mean(full_cov_t, axis=1))
            if t > 0:
                treat_level += (2.0 * A[:, t - 1] - 1.0)
            treat_level = np.clip(treat_level, 0.0, float(treat_level_max))

            base_term = err_Y[:, t]
            past_term = 0.0
            for i in range(h):
                if t - i >= 0:
                    base_part = np.mean(base_vitals[:, t - i, 0:5], axis=1)
                    rest_part = np.mean(np.concatenate([base_vitals[:, t - i, 5:], Z[:, t - i, :]], axis=1), axis=1)
                    past_term += coef_ya[i] * np.tanh(np.sin(base_part) * A[:, t - i] + np.cos(rest_part) * A[:, t - i])

            Y[:, t] = 5.0 * past_term + base_term

            if t < T - 1:
                err_Z = np.random.normal(loc=0.0, scale=config["noise_Z"], size=(n, config["num_z"]))
                Z[:, t + 1, :] = _update_z(Z[:, t, :], A[:, t], base_vitals[:, t, :], Y[:, t], err_Z)

        full_vitals = np.concatenate([base_vitals, Z], axis=2)
        return A, Y, full_vitals

    A_f, Y_f, full_vitals_f = generate_data_factual()
    A_cf, Y_cf, full_vitals_cf = generate_data_counterfactual_with_policy()

    def create_dataset(Y, A, full_vitals):
        data = np.concatenate((np.expand_dims(Y, 2), np.expand_dims(A, 2), full_vitals), axis=2)
        Y_prev = Y.copy()
        Y_prev[:, 0] = np.zeros(n)
        Y_prev[:, 1:] = Y[:, 0 : (T - 1)]
        data = np.concatenate((data, np.expand_dims(Y_prev, 2)), 2)
        return data

    return create_dataset(Y_f, A_f, full_vitals_f), create_dataset(Y_cf, A_cf, full_vitals_cf)


# Kept for parity with mimic_extract; unused by pipelines for the semi-synthetic setup.
def load_mimic_covariates(n, T, vital_list, static_list, mimic_extract_file, seed):
    h5 = pd.HDFStore(mimic_extract_file, "r")

    all_vitals = h5["/vitals_labs_mean"][vital_list].copy()
    all_vitals = all_vitals.droplevel(["hadm_id", "icustay_id"])

    column_names = []
    for column in all_vitals.columns:
        if isinstance(column, str):
            column_names.append(column)
        else:
            column_names.append(column[0])
    all_vitals.columns = column_names

    # Filling NA
    all_vitals = all_vitals.fillna(method="ffill")
    all_vitals = all_vitals.fillna(method="bfill")

    # Static features
    static_features = None
    if static_list is not None:
        static_features = h5[static_list].copy()
        static_features = static_features.droplevel(["hadm_id", "icustay_id"])

    # Filtering out users with time length < 2T
    user_sizes = all_vitals.groupby("subject_id").size()
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

    vitals_grouped = all_vitals.groupby("subject_id")
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


