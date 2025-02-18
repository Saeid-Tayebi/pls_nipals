import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import f
from scipy.stats import chi2


class pcaSkleanr:
    def __init__(self, X, Ncom):
        self.model = PCA(n_components=Ncom).fit(X=X)
        self.T = self.model.transform(X=X)
        self.P = self.model.components_.T
        self.T2 = np.sum(self.T**2 / np.var(self.T, axis=0, ddof=1), axis=1)
        n_samples = X.shape[0]
        self.T2lim = (Ncom * (n_samples - 1) / (n_samples - Ncom)
                      ) * f.ppf(0.95, Ncom, n_samples - Ncom)
        self.xhat = self.T @ self.P.T
        self.SPE = np.sum((X - self.xhat) ** 2, axis=1)  # Compute SPE

        mean_SPE = np.mean(self.SPE)
        var_SPE = np.var(self.SPE, ddof=1)
        self.SPE_limit = var_SPE / (2 * mean_SPE) * chi2.ppf(0.95,
                                                             df=2 * mean_SPE ** 2 / (var_SPE + 1e-15))

    def eval(self, xtes):
        pls: PCA = self.model
        Ttes = pls.transform(xtes)
        xhat = Ttes @ pls.components_  # Reconstructed X
        SPEtes = np.sum((xtes - xhat) ** 2, axis=1)  # Compute SPE

        std_T = np.std(self.T, axis=0, ddof=1)
        # Compute Hotelling’s T²
        T2tes = np.sum((Ttes / std_T) ** 2, axis=1)

        return xhat, Ttes, T2tes, SPEtes
