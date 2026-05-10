import numpy as np
import torch
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import rbf_kernel

class RandomFourierFeatures:
    """
    Random Fourier Features for approximating RBF kernel
    Converts joint distribution of (H_t, A_t) into finite-dimensional embeddings
    """
    
    def __init__(self, input_dim, n_features=128, gamma=1.0, random_state=42):
        self.input_dim = input_dim
        self.n_features = n_features
        self.gamma = gamma
        
        np.random.seed(random_state)
        self.W = np.random.normal(0, np.sqrt(2 * gamma), (input_dim, n_features))
        self.b = np.random.uniform(0, 2 * np.pi, n_features)
    
    def transform(self, X):
        """Transform features to RFF space"""
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        projections = np.dot(X, self.W) + self.b
        return np.sqrt(2.0 / self.n_features) * np.cos(projections)


class MMDKernelMDS:
    """MMD (Maximum Mean Discrepancy) + MDS for policy embedding"""
    
    def __init__(self, gamma=1.0, n_components=16, random_state=42):
        """
        Args:
            gamma: Gaussian kernel parameter (higher = more local)
            n_components: Number of MDS dimensions
            random_state: Random seed
        """
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state
        self.mds = MDS(n_components=n_components, dissimilarity='precomputed', 
                      random_state=random_state, normalized_stress='auto', n_init=4)
    
    def compute_mmd_distance(self, features_A, features_B):
        """
        Compute MMD distance between two sets of samples using Gaussian kernel
        
        Args:
            features_A: Array of shape (n_samples_A, n_features)
            features_B: Array of shape (n_samples_B, n_features)
            
        Returns:
            mmd_distance: Scalar MMD distance
        """
        if len(features_A) == 0 or len(features_B) == 0:
            return 0.0
            
        # Compute kernel matrices
        K_AA = rbf_kernel(features_A, features_A, gamma=self.gamma)
        K_BB = rbf_kernel(features_B, features_B, gamma=self.gamma)  
        K_AB = rbf_kernel(features_A, features_B, gamma=self.gamma)
        
        # MMD² = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
        mmd_squared = K_AA.mean() + K_BB.mean() - 2 * K_AB.mean()
        
        # Ensure non-negative (numerical precision issues)
        return np.sqrt(max(0, mmd_squared))
    
    def compute_mmd_distance_matrix(self, policy_features_list):
        """
        Compute pairwise MMD distances between multiple policies
        
        Args:
            policy_features_list: List of arrays, each containing samples for one policy
            
        Returns:
            distance_matrix: Array of shape (n_policies, n_policies)
        """
        n_policies = len(policy_features_list)
        distance_matrix = np.zeros((n_policies, n_policies))
        
        for i in range(n_policies):
            for j in range(i, n_policies):
                if i == j:
                    distance_matrix[i, j] = 0.0
                else:
                    dist = self.compute_mmd_distance(
                        policy_features_list[i], 
                        policy_features_list[j]
                    )
                    dist = 0 if dist < 1e-5 else dist # ensure non-negative and numerical stability
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist  # Symmetric
        
        return distance_matrix
    
    def fit_transform(self, policy_features_list):
        """
        Compute MMD distances and apply MDS
        
        Args:
            policy_features_list: List of arrays, each containing samples for one policy
            
        Returns:
            coordinates: Array of shape (n_policies, n_components)
            distance_matrix: Array of shape (n_policies, n_policies)
        """
        distance_matrix = self.compute_mmd_distance_matrix(policy_features_list)
        if np.max(distance_matrix) > 0:
            distance_matrix = distance_matrix / np.max(distance_matrix) # normalize to [0, 1]
        coordinates = self.mds.fit_transform(distance_matrix)
        return coordinates, distance_matrix


def create_policy_features(dataset, max_time_steps=10,start_time=0):
    """
    Create policy features from observed trajectories and counterfactual policy
    
    Args:
        dataset: Dataset with trajectories
        max_time_steps: Maximum time steps
        
    Returns:
        features_by_time: List of arrays, each of shape (n_samples, feature_dim) for each time step
    """
    features_by_time = []
    
    for t in range(start_time, max_time_steps):
        features_t = []
        
        for idx in range(len(dataset)):
            sample = dataset[idx]
            L = sample['L']
            A = sample['A']
            W = sample['W']
            a = sample['a']
            # L = sample['L'].squeeze()
            # A = sample['A'].squeeze()
            # W = sample['W'].squeeze()
            # a = sample['a'].squeeze()
            
            if t < len(L):
                # Create history H_t = (W, L[:t+1], A[:t]) with padding
                W_flat = W.flatten() if W.dim() > 0 else W.unsqueeze(0) # NOTE: here
                # L_padded = torch.zeros(max_time_steps, L_dim)
                # L_padded[:t+1] = L[:t+1]
                # A_padded = torch.zeros(max_time_steps-1, A_dim)
                # if t > 0:
                #     A_padded[:t,:] = A[:t,:]
                if t > 0:
                    H_t = torch.cat([W_flat, L[:t+1].flatten(), A[:t,:].flatten()])
                else:
                    H_t = torch.cat([W_flat, L[:t+1].flatten()])
                
                # Joint feature (H_t, A_t)
                joint_feature = torch.cat([H_t, a[t-start_time]])
                features_t.append(joint_feature.numpy())
        
        if features_t:
            features_by_time.append(np.array(features_t))
        else:
            features_by_time.append(np.array([]).reshape(0, -1))
    
    return features_by_time


def policy_embed_rff(features_t, rff_module):
    """
    Convert policy features to RFF features
    
    Args:
        features_t: Array of shape (n_samples, feature_dim) for a specific time step
        rff_module: RFF transformer
        
    Returns:
        rff_features: Array of shape (n_samples, n_rff_features)
    """
    if len(features_t) == 0:
        return np.array([]).reshape(0, rff_module.n_features)
    
    return rff_module.transform(features_t)


def compute_mds_coordinates_per_timestep_mmd(policy_features_by_time, gamma=1, n_components=16):
    """
    Compute MDS coordinates for policies at each time step using MMD distances
    
    Args:
        policy_features_by_time: Dict mapping policy_id -> list of feature arrays
                                Each policy_id maps to a list where index t contains 
                                feature array of shape (n_samples, n_features) for time step t
        gamma: Base multiplier for the per-timestep median-heuristic gamma_t.
        n_components: Number of MDS dimensions
        
    Returns:
        results: Dict with per-time-step MDS results:
            - coordinates_by_time: list of dicts (or None) with coordinates/policy_ids
            - distance_matrices: list of pairwise MMD distance matrices (or None)
            - stress_values: list of MDS stress values (or None)
            - gamma_by_time: list of gamma_t used at each time step (or None)
    """
    def _median_heuristic_gamma(policy_features_list, max_samples: int = 512, random_state: int = 42, fallback_gamma: float = 1.0):
        """
        Median heuristic for RBF kernel:
            gamma = 1 / (2 * median(||x - x'||^2))
        Computed on a pooled subsample across all policies at a single time step.
        """
        if len(policy_features_list) == 0:
            return fallback_gamma
        X = np.concatenate(policy_features_list, axis=0)
        if X.shape[0] < 2:
            return fallback_gamma

        rng = np.random.default_rng(random_state)
        if X.shape[0] > max_samples:
            idx = rng.choice(X.shape[0], size=max_samples, replace=False)
            X = X[idx]

        # Pairwise squared Euclidean distances
        G = np.sum(X * X, axis=1, keepdims=True)
        dist2 = G + G.T - 2.0 * (X @ X.T)
        dist2 = np.maximum(dist2, 0.0)

        tri = dist2[np.triu_indices(dist2.shape[0], k=1)]
        tri = tri[np.isfinite(tri)]
        tri = tri[tri > 0]
        if tri.size == 0:
            return fallback_gamma

        med = float(np.median(tri))
        if not np.isfinite(med) or med <= 0:
            return fallback_gamma
        return 1.0 / (2.0 * med)

    policy_ids = sorted(policy_features_by_time.keys())
    max_time_steps = len(policy_features_by_time[policy_ids[0]])
    
    results = {}
    
    results = {
        'coordinates_by_time': [],
        'distance_matrices': [],
        'stress_values': [],
        'gamma_by_time': [],
    }
    
    for t in range(max_time_steps):
        print(f"  Time step {t}")
        
        # Collect all feature arrays for each policy at time t
        policy_features_list = []
        valid_policy_ids = []
        
        for policy_id in policy_ids:
            features_t = policy_features_by_time[policy_id][t]
            if len(features_t) > 0:
                # Keep all samples for this policy (no averaging!)
                policy_features_list.append(features_t)
                valid_policy_ids.append(policy_id)
        
        if len(policy_features_list) > 1:
            gamma_t = gamma * _median_heuristic_gamma(
                policy_features_list,
                max_samples=512,
                random_state=42 + t,
                fallback_gamma=1.0,
            )
            mmd_mds = MMDKernelMDS(gamma=gamma_t, n_components=n_components)
            # Apply MMD + MDS using all samples
            coordinates, distance_matrix = mmd_mds.fit_transform(policy_features_list)
            
            results['coordinates_by_time'].append({
                'coordinates': coordinates,
                'policy_ids': valid_policy_ids
            })
            results['distance_matrices'].append(distance_matrix)
            results['stress_values'].append(mmd_mds.mds.stress_)
            results['gamma_by_time'].append(gamma_t)
            
            # Print sample sizes for each policy
            sample_sizes = [len(features) for features in policy_features_list]
            print(f"    Processed {len(valid_policy_ids)} policies with sample sizes {sample_sizes}")
            print(f"    gamma_t (median heuristic): {gamma_t:.6g}")
            print(f"    MDS stress: {mmd_mds.mds.stress_:.4f}")
        else:
            print(f"    Insufficient data for time step {t}")
            results['coordinates_by_time'].append(None)
            results['distance_matrices'].append(None)
            results['stress_values'].append(None)
            results['gamma_by_time'].append(None)

    return results


def policy_embed_mmd_mds(policy_features_list, n_components=16, gamma=1.0, mds_module=None):
    """
    Convert policy features to MDS coordinates using MMD distances
    
    Args:
        policy_features_list: List of arrays, each containing samples for one policy
        n_components: Number of MDS dimensions
        gamma: Gaussian kernel parameter
        mds_module: Optional pre-initialized MMDKernelMDS instance for reuse
        
    Returns:
        coordinates: Array of shape (n_policies, n_components)
        distance_matrix: Array of shape (n_policies, n_policies)
    """
    if len(policy_features_list) == 0:
        return np.array([]).reshape(0, n_components), np.array([]).reshape(0, 0)
    
    if len(policy_features_list) == 1:
        # Single policy case - return zero coordinates
        return np.zeros((1, n_components)), np.zeros((1, 1))
    
    # Use provided module or create new one
    if mds_module is None:
        mds_module = MMDKernelMDS(gamma=gamma, n_components=n_components)
    
    return mds_module.fit_transform(policy_features_list)



