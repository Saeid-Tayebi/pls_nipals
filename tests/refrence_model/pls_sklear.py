import numpy as np
from sklearn.cross_decomposition import PLSRegression
from scipy.stats import f
from scipy.stats import chi2


class plsSkleanr:
    def __init__(self, X, Y, Ncom):
        self.model = PLSRegression(n_components=Ncom).fit(X, Y)
        self.T = self.model.x_scores_
        self.P = self.model.x_loadings_
        self.Q = self.model.y_loadings_
        self.U = self.model.y_scores_
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
        pls: PLSRegression = self.model
        Ttes = pls.transform(xtes)
        xhat = Ttes @ pls.x_loadings_.T  # Reconstructed X
        SPEtes = np.sum((xtes - xhat) ** 2, axis=1)  # Compute SPE

        std_T = np.std(pls.x_scores_, axis=0, ddof=1)
        # Compute Hotelling’s T²
        T2tes = np.sum((Ttes / std_T) ** 2, axis=1)

        ypre = pls.predict(xtes)

        return ypre, Ttes, T2tes, SPEtes
