import numpy as np
from scipy.linalg import toeplitz
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class MeanRevertingPortfolio:
    def __init__(self):
        pass
    
    def create_transformation_matrix(self, pairs_data):
        """
        Create B matrix (M x N) mapping N pairs to M assets
        
        Parameters:
        - pairs_data: List of pairs with asset names
        
        Returns:
        - B: Transformation matrix (M x N)
        - asset_names: Ordered list of assets
        - pair_names: Ordered list of pairs
        """
        # Get all unique assets and pairs
        all_assets = sorted({a for pair in pairs_data for a in [pair[0], pair[1]]})
        pair_names = [f"{pair[0]}-{pair[1]}" for pair in pairs_data]
        
        M = len(all_assets)  # Number of assets
        N = len(pairs_data)  # Number of pairs
        
        # Create asset to index mapping
        asset_to_idx = {a:i for i,a in enumerate(all_assets)}
        
        # Initialize B matrix (M x N)
        B = np.zeros((M, N))
        
        for j, pair in enumerate(pairs_data):
            y_asset, x_asset = pair[0], pair[1]
            beta = pair[2]['coint_coef']
            
            # Spread = y - βx
            B[asset_to_idx[y_asset], j] = 1      # y asset
            B[asset_to_idx[x_asset], j] = -beta  # x asset
        
        return B, all_assets, pair_names

    def generate_matrices(self, pairs_data, max_lag=5):
        """
        Compute lag-i auto-covariance matrices (Mi) from pairs spread data
        with proper B matrix transformation
        
        Parameters:
        - pairs_data: List of pairs with spread data
        - max_lag: Maximum lag order (p) to consider
        
        Returns:
        - M_matrices: List of M0, M1, ..., Mp auto-covariance matrices
        - B: Transformation matrix
        - asset_names: Ordered list of asset names
        """
        # Create transformation matrix
        B, all_assets, pair_names = self.create_transformation_matrix(pairs_data)
        num_pairs = len(pairs_data)
        
        # Get all spread series (already transformed)
        spreads = [pair[2]['spread'].values for pair in pairs_data]
        spread_matrix = np.vstack(spreads).T  # shape: (T, n) where T is time length, n is num pairs
        
        # Center the data (subtract mean)
        spread_matrix = spread_matrix - np.mean(spread_matrix, axis=0)
        
        # Compute auto-covariance matrices
        M_matrices = []
        
        for lag in range(max_lag + 1):
            if lag == 0:
                # M0 is the contemporaneous covariance matrix
                Mi = np.cov(spread_matrix, rowvar=False)
            else:
                # For lags > 0, compute cross-products of lagged series
                Mi = np.zeros((num_pairs, num_pairs))
                for t in range(lag, spread_matrix.shape[0]):
                    Mi += np.outer(spread_matrix[t], spread_matrix[t - lag])
                Mi /= (spread_matrix.shape[0] - lag)
            
            M_matrices.append(Mi)
        
        return M_matrices, B, all_assets, pair_names

    def U(self, w, M_matrices, xi=1.0, zeta=1.0, eta=1.0):
        """Compute U(w) as given in the problem."""
        M0 = M_matrices[0]
        M1 = M_matrices[1]
        term1 = xi * (w.T @ M0 @ w) / (w.T @ M0 @ w)
        term2 = zeta * ((w.T @ M1 @ w) / (w.T @ M0 @ w)) ** 2
        term3 = eta * sum([((w.T @ M_matrices[i] @ w) / (w.T @ M0 @ w)) ** 2 for i in range(2, len(M_matrices))])
        return term1 + term2 + term3

    def R(self, w, M0, k_coeffs=[1.0]):
        """Compute R(w) (regularization term)."""
        denom = sum(k * (w.T @ M0 @ w) ** n for n, k in enumerate(k_coeffs, 1))
        return 1.0 / denom

    def S(self, w, B, epsilon=1e-3):
        """Compute the smoothed sparsity term S(w)."""
        Bw = B @ w
        return np.sum(1 - np.exp(-(np.abs(Bw) ** 2) / epsilon))

    def objective(self, w, M_matrices, B, xi, zeta, eta, rho, gamma, epsilon, k_coeffs):
        """Compute the full objective function f(w)."""
        M0 = M_matrices[0]
        return self.U(w, M_matrices, xi, zeta, eta) + rho * self.R(w, M0, k_coeffs) + gamma * self.S(w, B, epsilon)

    def surrogate_U(self, w, w_k, M_matrices, xi, zeta, eta):
        """Majorizing surrogate for U(w)."""
        M0 = M_matrices[0]
        denom_k = w_k.T @ M0 @ w_k
        # Linear approximation of 1/(w^T M0 w)
        inv_denom_approx = 1 / denom_k - (2 * M0 @ w_k).T @ (w - w_k) / (denom_k ** 2)
        # Approximate U(w) using the linearized denominator
        U_approx = xi * (w.T @ M0 @ w) * inv_denom_approx + \
                   zeta * ((w.T @ M_matrices[1] @ w) * inv_denom_approx) ** 2 + \
                   eta * sum([((w.T @ M_i @ w) * inv_denom_approx) ** 2 for M_i in M_matrices[2:]])
        return U_approx

    def surrogate_R(self, w, w_k, M0, k_coeffs):
        """Majorizing surrogate for R(w)."""
        denom_k = sum(k * (w_k.T @ M0 @ w_k) ** n for n, k in enumerate(k_coeffs, 1))
        grad_R = -2 * M0 @ w_k * sum(n * k * (w_k.T @ M0 @ w_k) ** (n-1) for n, k in enumerate(k_coeffs, 1)) / (denom_k ** 2)
        return self.R(w_k, M0, k_coeffs) + grad_R.T @ (w - w_k)

    def surrogate_S(self, w, w_k, B, epsilon):
        """Majorizing surrogate for S(w)."""
        Bw_k = B @ w_k
        g_k = 1 - np.exp(-(np.abs(Bw_k) ** 2) / epsilon)
        grad_g = (2 / epsilon) * (B.T @ (Bw_k * np.exp(-(np.abs(Bw_k) ** 2) / epsilon)))
        return self.S(w_k, B, epsilon) + grad_g.T @ (w - w_k)

    def total_surrogate(self, w, w_k, M_matrices, B, xi, zeta, eta, rho, gamma, epsilon, k_coeffs):
        """Combined surrogate function Q(w | w_k)."""
        M0 = M_matrices[0]
        Q_U = self.surrogate_U(w, w_k, M_matrices, xi, zeta, eta)
        Q_R = self.surrogate_R(w, w_k, M0, k_coeffs)
        Q_S = self.surrogate_S(w, w_k, B, epsilon)
        return Q_U + rho * Q_R + gamma * Q_S

    def projected_gradient_descent(self, w_init, M_matrices, B, params, max_iter=100, tol=1e-6):
        """Projected Gradient Descent with convergence tracking."""
        w_k = w_init.copy()
        history = {
            'U': [],
            'R': [],
            'S': [],
            'f': [],
            'w_norm': []
        }
        
        for _ in range(max_iter):
            # Compute current objective components
            U_val = self.U(w_k, M_matrices, params['xi'], params['zeta'], params['eta'])
            R_val = self.R(w_k, M_matrices[0], params['k_coeffs'])
            S_val = self.S(w_k, B, params['epsilon'])
            f_val = U_val + params['rho'] * R_val + params['gamma'] * S_val
            
            # Store values
            history['U'].append(U_val)
            history['R'].append(R_val)
            history['S'].append(S_val)
            history['f'].append(f_val)
            history['w_norm'].append(np.linalg.norm(w_k))
            
            # Minimize the surrogate
            res = minimize(
                fun=lambda w: self.total_surrogate(w, w_k, M_matrices, B, **params),
                x0=w_k,
                method='L-BFGS-B',
            )
            w_new = res.x
            
            # Check convergence
            if np.linalg.norm(w_new - w_k) < tol:
                break
            w_k = w_new
        
        return w_k, history

    def optimize_pair_weights(self, pairs_data, params=None, max_lag=5, max_iter=100, tol=1e-6, min_weight=0.01):
        """Optimize pair weights with clean formatting
        
        Parameters:
        - min_weight: Minimum weight threshold (weights below this will be set to 0)
        """
        if params is None:
            params = {
                'xi': 1.0,
                'zeta': 1.0,
                'eta': 1.0,
                'rho': 0.1,
                'gamma': 0.1,
                'epsilon': 1e-3,
                'k_coeffs': [1.0, 0.5],
            }
        
        M_matrices, B, asset_names, pair_names = self.generate_matrices(pairs_data, max_lag)
        w_init = np.ones(len(pairs_data)) / len(pairs_data)
        w_opt, history = self.projected_gradient_descent(w_init, M_matrices, B, params, max_iter, tol)
        
        # Process weights
        weights = {}
        for pair, weight in zip(pair_names, w_opt):
            # Convert pair name format and store raw weight
            weights[pair.replace('-', '_')] = float(weight)
        
        # Normalize by sum of absolute values
        total = sum(abs(w) for w in weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        # Clean up weights
        cleaned_weights = {}
        for pair, weight in weights.items():
            # 1. Round to 4 decimal places
            rounded = round(weight, 4)
            
            # 2. Apply minimum weight threshold
            if abs(rounded) < min_weight:
                rounded = 0.0
            
            # 3. Remove floating-point artifacts for clean display
            cleaned_weights[pair] = float(f"{rounded:.4f}")
        
        # Re-normalize after cleaning (optional)
        total_clean = sum(abs(w) for w in cleaned_weights.values())
        if total_clean > 0:
            cleaned_weights = {k: round(100 * v/total_clean, 4) for k, v in cleaned_weights.items()}
            
        
        return {
            'weights': cleaned_weights,
            'optimal_w': w_opt,
            'history': history,
            'pair_names': pair_names,
            'asset_names': asset_names
        }

    def get_strategy_weights(self, pairs_data, normalize=True, **kwargs):
        """Get weights ready for summarize_results"""
        result = self.optimize_pair_weights(pairs_data, **kwargs)
        weights = result['weights']
        
        return weights, result
