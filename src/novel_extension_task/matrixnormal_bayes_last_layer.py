
import numpy as np
import torch

# SciPy is used only for the Sylvester solve (mean update for MFVI + analytic mean)
from scipy.linalg import solve_sylvester


def make_v0(a: float, b: float, D: int, device=None, dtype=torch.float32):
    """V0 = a I + b 11^T"""
    if device is None:
        device = "cpu"
    I = torch.eye(D, device=device, dtype=dtype)
    ones = torch.ones((D, 1), device=device, dtype=dtype)
    V0 = a * I + b * (ones @ ones.T)
    return V0


def inv_v0_aI_b11T(a: float, b: float, D: int, device=None, dtype=torch.float32):
    """Closed-form inverse for V0 = a I + b 11^T (Sherman-Morrison).
    Assumes SPD: a > 0 and a + bD > 0.
    """
    if device is None:
        device = "cpu"

    if not (a > 0.0 and (a + b * D) > 0.0):
        raise ValueError(f"V0 not SPD with a={a}, b={b}, D={D}. Need a>0 and a+bD>0.")

    I = torch.eye(D, device=device, dtype=dtype)
    ones = torch.ones((D, 1), device=device, dtype=dtype)

    alpha = 1.0 / a
    beta = b / (a * (a + b * D))
    V0_inv = alpha * I - beta * (ones @ ones.T)
    return V0_inv


def _to_numpy(x: torch.Tensor):
    return x.detach().cpu().double().numpy()


def _to_torch(x_np, device, dtype):
    return torch.from_numpy(np.asarray(x_np)).to(device=device, dtype=dtype)


class MatrixNormalBayesianLastLayer:
    """
    Bayesian regression head on top of frozen features Phi.
    y | W, phi ~ N(W^T phi, sigma2 I_D)
    Prior: W ~ MN(0, s2 I_K, V0) with V0 = a I + b 11^T (structured correlations)
    """

    def __init__(self, sigma2=1.0, s2=1.0, a=1.0, b=0.0):
        self.sigma2 = float(sigma2)
        self.s2 = float(s2)
        self.a = float(a)
        self.b = float(b)

    # ANALYTIC (mean + fast predictive cov via eig trick) 

    def fit_analytic(self, Phi: torch.Tensor, Y: torch.Tensor):
        """
        Returns parameters needed for predictions and acquisition.
        We keep it "analytic" (exact) for mean, and predictive covariance is computed
        using eigendecompositions (no big KD inversion).
        """
        device, dtype = Phi.device, Phi.dtype
        N, K = Phi.shape
        D = Y.shape[1]

        # Precompute G and C
        G = Phi.T @ Phi  # (K,K)
        C = Phi.T @ Y    # (K,D)

        # V0^{-1} and eigendecomp
        V0_inv = inv_v0_aI_b11T(self.a, self.b, D, device=device, dtype=dtype)

        # Mean M solves: (1/sigma2) G M + (1/s2) M V0_inv = (1/sigma2) C
        A = (1.0 / self.sigma2) * G
        B = (1.0 / self.s2) * V0_inv
        RHS = (1.0 / self.sigma2) * C

        M_np = solve_sylvester(_to_numpy(A), _to_numpy(B), _to_numpy(RHS))
        M = _to_torch(M_np, device, dtype)

        # Eigendecompositions for predictive covariance
        # G is symmetric PSD, V0_inv is symmetric SPD
        evals_G, Q = torch.linalg.eigh(G)          # evals_G: (K,), Q: (K,K)
        evals_Vinv, P = torch.linalg.eigh(V0_inv)  # evals_Vinv: (D,), P: (D,D)

        params = {
            "type": "analytic",
            "M": M,
            "G_evals": evals_G,
            "G_evecs": Q,
            "Vinv_evals": evals_Vinv,
            "Vinv_evecs": P,  # orthonormal
        }
        return params

    def _analytic_predictive_s_diag(self, Phi_star: torch.Tensor, params):
        """
        Computes s(x) in the eigenbasis of V0_inv:
        Cov_epistemic(y_tilde) = diag(s), where y = P y_tilde
        and P are eigenvectors of V0_inv.
        """
        device, dtype = Phi_star.device, Phi_star.dtype
        Q = params["G_evecs"]
        lam = params["G_evals"].clamp(min=0.0)  # numerical safety
        gam = params["Vinv_evals"].clamp(min=1e-12)

        # alpha_{k,d} = 1 / ( (lam_k/sigma2) + (gam_d/s2) )
        alpha = 1.0 / ((lam.view(-1, 1) / self.sigma2) + (gam.view(1, -1) / self.s2))  # (K,D)

        # phi_tilde = Q^T phi  => for batch: Phi_star @ Q
        phi_tilde = Phi_star @ Q  # (B,K)
        phi_sq = phi_tilde * phi_tilde

        s = phi_sq @ alpha  # (B,D)
        return s

    def predictive(self, Phi_star: torch.Tensor, params, return_cov=False):
        single = False
        if Phi_star.ndim == 1:
            Phi_star = Phi_star.unsqueeze(0)
            single = True

        M = params["M"]
        mean = Phi_star @ M  # (B,D)

        if not return_cov:
            return mean.squeeze(0) if single else mean, None

        D = mean.shape[1]
        P = params["Vinv_evecs"]

        s = self._analytic_predictive_s_diag(Phi_star, params)  # (B,D)
        # Cov(y) = sigma2 I + P diag(s) P^T. Since P is orthonormal, eigenvalues are sigma2 + s_d.
        covs = []
        I = torch.eye(D, device=Phi_star.device, dtype=Phi_star.dtype)
        for i in range(Phi_star.shape[0]):
            Sd = torch.diag(s[i])
            cov = self.sigma2 * I + P @ Sd @ P.T
            covs.append(cov)
        cov = torch.stack(covs, dim=0)  # (B,D,D)

        if single:
            return mean.squeeze(0), cov.squeeze(0)
        return mean, cov

    def acquisition_score(self, Phi_star: torch.Tensor, params, mode="trace_total"):
        """
        Scalarizations for analytic case are very cheap because in the P-basis
        the covariance is diagonal with entries sigma2 + s_d.
        """
        if Phi_star.ndim == 1:
            Phi_star = Phi_star.unsqueeze(0)
        s = self._analytic_predictive_s_diag(Phi_star, params)  # (B,D)
        D = s.shape[1]

        if mode == "trace_total":
            return self.sigma2 * D + s.sum(dim=1)

        if mode == "trace_epistemic":
            return s.sum(dim=1)

        if mode == "logdet_total":
            return torch.log(self.sigma2 + s).sum(dim=1)

        if mode == "logdet_epistemic":
            # logdet of P diag(s) P^T is sum log(s), but s can be tiny early on
            return torch.log(s.clamp(min=1e-12)).sum(dim=1)
        
        if mode == "trace_epistemic_norm":
            # epistemic trace / ||phi||^2
            norm2 = (Phi_star * Phi_star).sum(dim=1) + 1e-8
            return s.sum(dim=1) / norm2

        if mode == "logdet_total_norm":
            # logdet(sigma2 I + Cov_epi) / ||phi||^2
            norm2 = (Phi_star * Phi_star).sum(dim=1) + 1e-8
            return torch.log(self.sigma2 + s).sum(dim=1) / norm2

        if mode == "logdet_epistemic_norm":
            # logdet(Cov_epi) / ||phi||^2 (Cov_epi = P diag(s) P^T so logdet = sum log s)
            norm2 = (Phi_star * Phi_star).sum(dim=1) + 1e-8
            return torch.log(s.clamp(min=1e-12)).sum(dim=1) / norm2


        raise ValueError(f"Unknown acquisition mode: {mode}")

    # ---------- MFVI (U diagonal, V full) ----------

    def fit_mfvi(self, Phi: torch.Tensor, Y: torch.Tensor, num_iters=10, eps=1e-8, verbose=False):
        """
        Coordinate-ascent MFVI:
            q(W) = MN(M, U, V)
        with U diagonal, V full SPD.
        Tracks parameter changes to diagnose convergence.
        """
        device, dtype = Phi.device, Phi.dtype
        N, K = Phi.shape
        D = Y.shape[1]

        G = Phi.T @ Phi  # (K,K)
        C = Phi.T @ Y    # (K,D)
        diagG = torch.diag(G)

        V0_inv = inv_v0_aI_b11T(self.a, self.b, D, device=device, dtype=dtype)

        # init: mean from ridge regression ignoring correlations (same as minimal extension)
        I_K = torch.eye(K, device=device, dtype=dtype)
        A_ridge = (G / self.sigma2) + (I_K / self.s2)
        M = torch.linalg.solve(A_ridge, C / self.sigma2)  # (K,D)

        # init U diag (classic MFVI diag approx)
        U_diag = 1.0 / torch.diag(A_ridge).clamp(min=eps)  # (K,)

        # init V = V0
        V = make_v0(self.a, self.b, D, device=device, dtype=dtype)

        prev_M = M.clone()
        prev_U_diag = U_diag.clone()
        prev_V = V.clone()

        for it in range(int(num_iters)):
            # M update: solve sylvester
            A = (1.0 / self.sigma2) * G
            B = (1.0 / self.s2) * V0_inv
            RHS = (1.0 / self.sigma2) * C

            M_np = solve_sylvester(_to_numpy(A), _to_numpy(B), _to_numpy(RHS))
            M = _to_torch(M_np, device, dtype)

            # U update (diagonal)
            tr_v = torch.trace(V0_inv @ V).clamp(min=eps)
            scale = float(D) / tr_v
            denom = (diagG / self.sigma2) + (1.0 / self.s2)
            U_diag = scale / denom.clamp(min=eps)

            # V update (full)
            tr_UG = (U_diag * diagG).sum()
            tr_U = U_diag.sum()

            mat = (tr_UG / self.sigma2) * torch.eye(D, device=device, dtype=dtype) + (tr_U / self.s2) * V0_inv
            V = float(K) * torch.linalg.inv(mat)

            # symmetrize a bit (numerical stability)
            V = 0.5 * (V + V.T)

            # Track relative parameter changes
            delta_M = (M - prev_M).norm() / (prev_M.norm() + eps)
            delta_U = (U_diag - prev_U_diag).norm() / (prev_U_diag.norm() + eps)
            delta_V = (V - prev_V).norm() / (prev_V.norm() + eps)
            if verbose:
                print(f"MFVI iter {it+1}: rel_change M={delta_M:.3e}, U_diag={delta_U:.3e}, V={delta_V:.3e}")

            prev_M = M.clone()
            prev_U_diag = U_diag.clone()
            prev_V = V.clone()

        params = {
            "type": "mfvi",
            "M": M,
            "U_diag": U_diag,
            "V": V,
        }
        return params

    def mfvi_predictive_stats(self, Phi_star: torch.Tensor, params):
        """
        For q(W)=MN(M,U,V) with U diag:
            mean = Phi_star @ M
            Cov_epistemic(y) = c(x) * V, with c(x) = phi^T U phi (scalar)
        """
        if Phi_star.ndim == 1:
            Phi_star = Phi_star.unsqueeze(0)

        M = params["M"]
        U_diag = params["U_diag"]
        V = params["V"]

        mean = Phi_star @ M  # (B,D)
        c = (Phi_star * Phi_star) @ U_diag  # (B,)
        return mean, c, V

    def acquisition_score_mfvi(self, Phi_star: torch.Tensor, params, mode="trace_total", eps=1e-12):
        if Phi_star.ndim == 1:
            Phi_star = Phi_star.unsqueeze(0)

        _, c, V = self.mfvi_predictive_stats(Phi_star, params)
        D = V.shape[0]

        if mode == "trace_total":
            return self.sigma2 * D + c * torch.trace(V)

        if mode == "trace_epistemic":
            return c * torch.trace(V)

        if mode in ["logdet_total", "logdet_epistemic"]:
            # Use eigenvalues of V (cheap, D=10). Then logdet(sigma2 I + c V) = sum log(sigma2 + c * eig(V))
            evals = torch.linalg.eigvalsh(V).clamp(min=eps)  # (D,)
            if mode == "logdet_epistemic":
                # logdet(c V) = D log(c) + logdet(V)
                c_safe = c.clamp(min=eps)
                return float(D) * torch.log(c_safe) + torch.log(evals).sum()
            return torch.log(self.sigma2 + c.view(-1, 1) * evals.view(1, -1)).sum(dim=1)
        
        if mode == "trace_epistemic_norm":
            # trace(Cov_epi) is c(x) * tr(V) but tr(V) is constant across x, so ranking is c(x)
            norm2 = (Phi_star * Phi_star).sum(dim=1) + 1e-8
            return c / norm2

        if mode == "logc_norm":
            # MFVI: logdet(Cov_epi) = D log c + logdet(V), logdet(V) constant across x
            norm2 = (Phi_star * Phi_star).sum(dim=1) + 1e-8
            return torch.log(c.clamp(min=1e-12)) / norm2

        if mode == "logdet_total_norm":
            # keep the "total logdet" version but normalize to reduce norm dominance
            evals = torch.linalg.eigvalsh(V).clamp(min=1e-12)  # (D,)
            total_logdet = torch.log(self.sigma2 + c.view(-1, 1) * evals.view(1, -1)).sum(dim=1)
            norm2 = (Phi_star * Phi_star).sum(dim=1) + 1e-8
            return total_logdet / norm2

        raise ValueError(f"Unknown acquisition mode: {mode}")

    def offdiag_mass(self, V: torch.Tensor):
        """A small diagnostic: relative off-diagonal energy of V."""
        diag = torch.diag(V)
        off = V - torch.diag(diag)
        return off.abs().sum() / (V.abs().sum() + 1e-8)
