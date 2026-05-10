import numpy as np
import pandas as pd
import atexit

from src.data.dataset_collection import SyntheticDatasetCollection_Simplified
from src.data.mimic_extract_complex.simulation import simulate_treat_outcomes, simulate_treatment_outcomes_function

class MIMICDataCache:
    """Singleton cache for MIMIC data to avoid reloading H5 file"""
    _instance = None
    _cache = {}
    _atexit_registered = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        if not cls._atexit_registered:
            atexit.register(cls.close_all)
            cls._atexit_registered = True
        return cls._instance

    def get_data(self, h5file_path):
        """Get cached DataFrame or load if not cached"""
        if h5file_path not in self._cache:
            print(f"Loading MIMIC data from {h5file_path} (first time)")
            self._cache[h5file_path] = pd.HDFStore(h5file_path, "r")
        else:
            print(f"Using cached MIMIC data for {h5file_path}")
        return self._cache[h5file_path]

    @classmethod
    def close_all(cls):
        """Close all open HDFStore handles and clear cache."""
        for path, store in list(cls._cache.items()):
            try:
                store.close()
            except Exception:
                pass
            finally:
                cls._cache.pop(path, None)

    def clear_cache(self):
        """Clear the cache (useful for testing or memory management)"""
        self.close_all()


class MIMICSynDatasetCollection(SyntheticDatasetCollection_Simplified):
    """
    Complex variant of the semi-synthetic MIMIC-Extract dataset.

    Compared to `src/data/mimic_extract`, the only change is that the simulator
    appends 5 additional time-varying variables that are *affected by actions*,
    yielding 15 time-varying variables total (10 real MIMIC vitals + 5 simulated).

    The returned raw array layout remains compatible with all existing adapters:
      raw[t] = [Y_t, A_t, vitals_t..., Y_prev_t]
    with vitals_t having dimension 15.
    """

    def __init__(self, num_patients, config, h5file_path, cf_treatment_sequence, seed=42, **kwargs):
        vital_list = [
            "heart rate",
            "sodium",
            "mean blood pressure",
            "glucose",
            "hematocrit",
            "respiratory rate",
            "prothrombin time pt",
            "hemoglobin",
            "creatinine",
            "blood urea nitrogen",
        ]

        self.seed = seed
        np.random.seed(seed)

        T = config["T"]

        cache = MIMICDataCache()
        df = cache.get_data(h5file_path)

        # Sample disjoint subject_ids for train/val/eval to avoid leakage across these splits.
        train_vitals, train_subject_ids = self._load_mimic_covariates_from_df(
            df, num_patients["train"], T, vital_list, None, seed=seed
        )
        factual_train, _ = simulate_treat_outcomes(
            {**config, "n": num_patients["train"]}, np.zeros((T,)), train_vitals, seed=seed
        )

        val_vitals, val_subject_ids = self._load_mimic_covariates_from_df(
            df, num_patients["val"], T, vital_list, None, seed=seed + 1000, exclude_subject_ids=train_subject_ids,
        )
        factual_val, _ = simulate_treat_outcomes(
            {**config, "n": num_patients["val"]}, np.zeros((T,)), val_vitals, seed=seed + 1000
        )

        eval_vitals, eval_subject_ids = self._load_mimic_covariates_from_df(
            df, num_patients["eval"], T, vital_list, None, seed=seed + 1500, exclude_subject_ids=np.concatenate([train_subject_ids, val_subject_ids]),
        )
        factual_eval, _ = simulate_treat_outcomes({**config, "n": num_patients["eval"]}, np.zeros((T,)), eval_vitals, seed=seed + 1500)

        # Test is only used to simulate Monte-Carlo ground truth, so overlap with other splits is acceptable.
        test_vitals, _ = self._load_mimic_covariates_from_df(df, num_patients["test"], T, vital_list, None, seed=seed + 2000)

        if callable(cf_treatment_sequence[0]):
            _, test_cf_treatment_seq = simulate_treatment_outcomes_function(
                {**config, "n": num_patients["test"]}, 
                cf_treatment_sequence, 
                test_vitals, 
                seed=seed + 2000
            )
        else:
            _, test_cf_treatment_seq = simulate_treat_outcomes(
                {**config, "n": num_patients["test"]}, 
                cf_treatment_sequence, 
                test_vitals, 
                seed=seed + 2000
            )

        self.train_f = factual_train
        self.val_f = factual_val
        self.eval_f = factual_eval
        self.test_cf_treatment_seq = test_cf_treatment_seq

        # Store data for efficient counterfactual generation
        self._df = df
        self._vital_list = vital_list
        self._config = config
        self._test_vitals = test_vitals
        self._T = T
        self._seed = seed

    def generate_counterfactual_test_data(self, new_cf_treatment_sequence):
        """
        Generate test data with a new counterfactual treatment sequence
        without reloading the H5 file.
        """
        if callable(new_cf_treatment_sequence[0]):
            _, new_test_cf_treatment_seq = simulate_treatment_outcomes_function(
                {**self._config, "n": self._test_vitals.shape[0]},
                new_cf_treatment_sequence,
                self._test_vitals,
                seed=self._seed + 2000,
            )
        else:
            _, new_test_cf_treatment_seq = simulate_treat_outcomes(
                {**self._config, "n": self._test_vitals.shape[0]},
                new_cf_treatment_sequence,
                self._test_vitals,
                seed=self._seed + 2000,
            )
        return new_test_cf_treatment_seq

    def _load_mimic_covariates_from_df(self, df, n, T, vital_list, static_list, seed, exclude_subject_ids=None):
        """
        Efficient version of load_mimic_covariates that works with pre-loaded DataFrame.
        Returns standardized base vitals of shape (n, T, 10).
        """
        from src.data.mimic_extract_complex.simulation import standardize_covariates

        all_vitals = df["/vitals_labs_mean"][vital_list].copy()
        all_vitals = all_vitals.droplevel(["hadm_id", "icustay_id"])

        column_names = []
        for column in all_vitals.columns:
            if isinstance(column, str):
                column_names.append(column)
            else:
                column_names.append(column[0])
        all_vitals.columns = column_names

        # Filling NA
        all_vitals = all_vitals.ffill()
        all_vitals = all_vitals.bfill()

        # Static features
        static_features = None
        if static_list is not None:
            static_features = df[static_list].copy()
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

        # Optionally exclude subject_ids to enforce disjoint splits.
        if exclude_subject_ids is not None and len(exclude_subject_ids) > 0:
            exclude_subject_ids = np.asarray(exclude_subject_ids)
            filtered_users = np.asarray([u for u in filtered_users if u not in set(exclude_subject_ids.tolist())])
        else:
            filtered_users = np.asarray(filtered_users)

        if n > len(filtered_users):
            raise ValueError(
                f"Not enough eligible subjects to sample n={n} after exclusion (eligible={len(filtered_users)})."
            )

        np.random.seed(seed)
        chosen_subject_ids = np.random.choice(filtered_users, size=n, replace=False)
        all_vitals = all_vitals.loc[chosen_subject_ids]

        vitals_grouped = all_vitals.groupby("subject_id")
        data_vitals = np.zeros((n, T, len(vital_list)))

        for i, cov in enumerate(vitals_grouped):
            arr = cov[1].to_numpy()
            for t in range(T):
                data_vitals[i, t, :] = arr[t, :]

        data_vitals = standardize_covariates(data_vitals)

        return data_vitals, chosen_subject_ids


