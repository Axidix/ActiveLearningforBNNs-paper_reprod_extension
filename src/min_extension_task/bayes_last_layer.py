import torch


class BayesianLastLayer:
    def __init__(self, sigma2=1.0, s2=1.0):
        self.sigma2 = sigma2       # noise variance 
        self.s2 = s2            # prior variance


    def fit_analytic(self, Phi, Y):
        """Phi : (N, K) features
        Y   : (N, D) one-hot regression targets"""
        N, K = Phi.shape
        I = torch.eye(K, device=Phi.device)

        # posterior covariance
        Sigma_inv = (Phi.T @ Phi) / self.sigma2 + I / self.s2
        Sigma = torch.linalg.inv(Sigma_inv)

        # posterior mean
        mu = Sigma @ (Phi.T @ Y) / self.sigma2

        return {"type": "analytic", "mu": mu, "Sigma": Sigma}


    def fit_mfvi(self, Phi, Y):
        """
        Returns optimal mean and diagonal covariance for MFVI posterior approximation.
        Correct: mean matches analytic, covariance is diagonal approx.
        """
        N, K = Phi.shape
        I = torch.eye(K, device=Phi.device)

        A = (Phi.T @ Phi) / self.sigma2 + I / self.s2         # (K,K)
        B = (Phi.T @ Y) / self.sigma2                        # (K,D)

        # MFVI mean = analytic mean (solve full system)
        M = torch.linalg.solve(A, B)                         # (K,D)

        # Diagonal covariance approximation (classic mean-field)
        S_diag = 1.0 / torch.diag(A)                         # (K,)

        return {"type": "mfvi", "M": M, "S_diag": S_diag}


    def predictive(self, Phi_star, params, return_cov=False):
        """Returns predictive mean and variance for inputs Phi_star.
        Returns covariance matrix if return_cov is True.
        """

        single = False
        if Phi_star.ndim == 1:
            Phi_star = Phi_star.unsqueeze(0)
            single = True

        if params["type"] == "analytic":
            mu, Sigma = params["mu"], params["Sigma"]
            mean = Phi_star @ mu
            tmp = Phi_star @ Sigma
            var = self.sigma2 + (tmp * Phi_star).sum(dim=1)
        elif params["type"] == "mfvi":
            M, S_diag = params["M"], params["S_diag"]
            mean = Phi_star @ M
            var = self.sigma2 + (Phi_star ** 2) @ S_diag
        else:
            raise ValueError(f"Unknown params type: {params.get('type', None)}")

        if single:
            mean = mean.squeeze(0)
            var = var.squeeze(0)

        if not return_cov:
            return mean, var

        D = mean.shape[-1]
        if single:
            cov = var * torch.eye(D, device=mean.device)
            return mean, var, cov

        I = torch.eye(D, device=mean.device).unsqueeze(0)
        cov = var.view(-1, 1, 1) * I
        return mean, var, cov
    

    def acquisition_score(self, Phi_star, params, mode="trace_total"):
        """Computes acquisition score based on predictive variance.
        Modes:
            'trace_total': uses total predictive variance (aleatoric + epistemic)
            'trace_epistemic': uses only epistemic variance (removes sigma2)
            'det': var**D
            'max': var
        """
        mean, var = self.predictive(Phi_star, params)  # mean: (B,D) or (D,), var: (B,) or scalar
        D = mean.shape[-1]

        if mode == "trace_total":
            return var * D

        if mode == "trace_epistemic":
            epi = var - self.sigma2
            epi = torch.clamp(epi, min=0.0)  # numerical safety
            return epi * D

        if mode == "trace_epistemic_norm":
            epi = var - self.sigma2
            epi = torch.clamp(epi, min=0.0)
            norm2 = (Phi_star ** 2).sum(dim=1) + 1e-8
            return (epi / norm2) * D

        if mode == "det":
            return var**D

        if mode == "max":
            return var

        raise ValueError(f"Unknown acquisition mode: {mode}")

