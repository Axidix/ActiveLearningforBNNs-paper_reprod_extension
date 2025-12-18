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

        return mu, Sigma


    def fit_mfvi(self, Phi, Y):
        """
        Returns optimal mean and diagonal covariance for MFVI posterior approximation.
        """
        # diagonal of Phi^T Phi
        Phi_sq = torch.sum(Phi ** 2, dim=0)

        # optimal diagonal covariance
        S_diag = 1.0 / (Phi_sq / self.sigma2 + 1.0 / self.s2)

        # optimal mean
        M = (Phi.T @ Y) / self.sigma2
        M = S_diag[:, None] * M

        return M, S_diag


    def predictive(self, Phi_star, params, return_cov=False):
        """Returns predictive mean and variance for inputs Phi_star.
        Returns covariance matrix if return_cov is True.
        """

        single = False
        if Phi_star.ndim == 1:
            Phi_star = Phi_star.unsqueeze(0)
            single = True

        # analytic
        if len(params) == 2 and params[1].ndim == 2:
            mu, Sigma = params
            mean = Phi_star @ mu
            # var = sigma2 + phi^T Sigma phi (batched)
            tmp = Phi_star @ Sigma
            var = self.sigma2 + (tmp * Phi_star).sum(dim=1)

        # mfvi
        else:
            M, S_diag = params
            mean = Phi_star @ M
            var = self.sigma2 + (S_diag * (Phi_star ** 2)).sum(dim=1)

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
    

    def acquisition_score(self, Phi_star, params, mode="trace"):
        """Computes acquisition score based on predictive variance."""
        
        mean, var = self.predictive(Phi_star, params)  # mean: (B,D) or (D,), var: (B,) or scalar

        D = mean.shape[-1]

        if mode == "trace":
            return var * D

        if mode == "det":
            return var**D

        if mode == "max":
            return var

        raise ValueError("Unknown acquisition mode")

