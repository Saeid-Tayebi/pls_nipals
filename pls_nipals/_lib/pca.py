import numpy as np
from scipy.stats import chi2, f
import matplotlib.pyplot as plt

# Output Classes


class pcaeval:
    def __init__(self, xhat, tscore, HT2, spe):
        self.xhat = xhat
        self.tscore = tscore
        self.HT2 = HT2
        self.spe = spe

# main class


class PcaClass:
    def __init__(self):
        # Initialize attributes if needed
        self.T = None
        self.P = None
        self.x_hat = None
        self.tsquared = None
        self.T2_lim = None
        self.ellipse_radius = None
        self.SPE_x = None
        self.SPE_lim_x = None
        self.Rsquared = None
        self.covered_var = None
        self.x_scaling = None
        self.Xtrain_normal = None
        self.Xtrain_scaled = None
        self.alpha = None
        self.n_component = None
# FIXME: check if the original data has a rwo with all zeroes, it should be deleted from X

    def fit(self, X: np.ndarray, n_component: int = None, alpha: float = 0.95, to_be_scaled: bool = True):
        """fit pca model using Nipals algorithm over x, alpha and scaling options are set to .95 and True respectively. if n_component is not provided, it will be determined by a eigenvalue_less_than_1 rule

        Args:
            X (np.ndarray): input data.
            n_component (int, optional): Number of PCA components. Defaults to None.
            alpha (float, optional): Modelling confidence limit. Defaults to 0.95.
            to_be_scaled (bool, optional): center and scales X and use it for all evaluation putposes. Defaults to True.

        Returns:
            pcamodel: a pca model with all methods.
        """
        # Preprocessing
        def preprocessing(X):
            # col with no variance
            removed_col = np.where(X.var(axis=0) < 1e-8)[0]
            X = np.delete(X, removed_col, axis=1) if removed_col.size > 0 else X
            # row with no variance
            removed_row = np.where(X.var(axis=1) < 1e-8)[0]
            X = np.delete(X, removed_row, axis=0) if removed_row.size > 0 else X
            return X, removed_row, removed_col

        # Data Preparation
        X, removed_row, removed_col = preprocessing(X)
        X_orining = X
        Cx = np.mean(X, axis=0)
        Sx = np.std(X, axis=0, ddof=1) + 1e-16
        if to_be_scaled == 1:
            X = (X - Cx) / Sx
        X, removed_row, removed_col = preprocessing(X)
        X_orining = np.delete(X_orining, removed_row, axis=0) if removed_row.size > 0 else X_orining
        X_orining = np.delete(X_orining, removed_col, axis=1) if removed_col.size > 0 else X_orining

        n_component = self.set_n_component(X, ploting=False) if n_component is None else n_component
        n_component = np.min((n_component, X.shape[1], X.shape[0]-1))

        if n_component < 1:
            raise ValueError("data does not have any variance")

        Num_obs = X.shape[0]
        K = X.shape[1]  # Num of X Variables
        X_0 = X

        # Blocks initialization
        T = np.zeros((Num_obs, n_component))
        P = np.zeros((K, n_component))
        covered_var = np.zeros((1, n_component))
        SPE_x = np.zeros_like(T)
        SPE_lim_x = np.zeros(n_component)
        tsquared = np.zeros_like(T)
        T2_lim = np.zeros(n_component)
        ellipse_radius = np.zeros(n_component)
        Rx = np.zeros(n_component)

        # NIPALS Algorithm
        for i in range(n_component):
            t1 = X[:, np.argmax(np.var(X_orining, axis=0, ddof=1))]
            while True:
                P1 = (t1.T @ X) / (t1.T @ t1)
                P1 = P1 / np.linalg.norm(P1)
                t_new = ((P1 @ X.T) / (P1.T @ P1)).T
                if np.allclose(t1, t_new, atol=1e-10):
                    break
                t1 = t_new
            x_hat = t1.reshape(-1, 1) @ P1.reshape(1, -1)
            X = X - x_hat
            P[:, i] = P1
            T[:, i] = t1

            covered_var[:, i] = np.var(t1, axis=0, ddof=1)
            # SPE_X
            SPE_x[:, i], SPE_lim_x[i], Rx[i] = self.SPE_calculation(
                T, P, X_0, alpha)

            # Hotelling T2 Related Calculations
            tsquared[:, i], T2_lim[i], ellipse_radius[i] = self.T2_calculations(
                T[:, : i + 1], i + 1, Num_obs, alpha)

        # Function Output
        self.T = T
        self.P = P
        self.x_hat = ((T @ P.T) * Sx) + Cx
        self.tsquared = tsquared
        self.T2_lim = T2_lim
        self.ellipse_radius = ellipse_radius
        self.SPE_x = SPE_x
        self.SPE_lim_x = SPE_lim_x
        self.Rsquared = Rx.T * 100
        self.covered_var = covered_var
        self.x_scaling = np.vstack((Cx, Sx))
        self.Xtrain_normal = X_orining
        self.Xtrain_scaled = X_0
        self.alpha = alpha
        self.n_component = n_component

        return self

    def set_n_component(self, Z, ploting=True) -> int:
        """return Num_com_sugg
        Z can be either X or Y
        determines the number of components that describe the data quite good using the eigenvalue_greater_than_one_rule


        Args:
            Z (_type_):input data
            ploting (bool, optional): ploting. Defaults to True.

        Returns:
            int: _description_
        """
        self.fit(Z, n_component=Z.shape[1])
        eig_val = self.covered_var
        Num_com_sugg = np.sum(eig_val > 1)
        if ploting == True:
            plt.figure()
            plt.bar(
                range(1, eig_val.shape[1] + 1),
                eig_val.reshape(-1),
                label="Covered Variance",
            )
            plt.xlabel("Components")
            plt.ylabel("Variance Covered")
            plt.plot([0, eig_val.shape[1] + 1], [1, 1],
                     "k--", label="Threshold Line")
            plt.legend()
            plt.show()

        return Num_com_sugg

    def evaluation(self, xtest: np.ndarray) -> pcaeval:
        """
        receive pca model and new observation and calculate its
        x_hat,T_score,Hotelin_T2,SPE_X

        Args:
            xtest (np.ndarray): new unseen datapoint

        Returns:
            pcaeval: include xhat, tscore,HotellingT2 and spe
        """
        xtest_scalled = self.scaler(xtest)

        tnew = xtest_scalled @ self.P
        xhat = tnew @ self.P.T

        SPE_X = np.sum((xtest_scalled - xhat)**2, axis=1)

        xhat = self.unscaler(xhat)
        Hotelin_T2 = np.sum(
            (tnew / np.std(self.T, axis=0, ddof=1)) ** 2, axis=1)
        return pcaeval(xhat, tnew, Hotelin_T2, SPE_X)

    def SPE_calculation(self, score, loading, Original_block, alpha):
        # Calculation of SPE and limits
        X_hat = score @ loading.T
        Error = Original_block - X_hat
        # Error.reshape(-1,loading.shape[1])
        spe = np.sum(Error**2, axis=1)
        spe_lim, Rsquare = None, None

        m = np.mean(spe)
        v = np.var(spe, ddof=1)
        spe_lim = v / (2 * m) * chi2.ppf(alpha, 2 * m**2 / (v + 1e-15))
        Rsquare = 1 - np.var(Error, ddof=1) / np.var(
            Original_block, ddof=1
        )  # not applicaple for pls vali
        return spe, spe_lim, Rsquare

    def T2_calculations(self, T, Num_com, Num_obs, alpha):
        # Calculation of Hotelling T2 statistics
        tsquared = np.sum((T / np.std(T, axis=0, ddof=1)) ** 2, axis=1)
        T2_lim = (
            (Num_com * (Num_obs**2 - 1))
            / (Num_obs * (Num_obs - Num_com))
            * f.ppf(alpha, Num_com, Num_obs - Num_com)
        )
        ellipse_radius = np.sqrt(
            T2_lim * np.std(T[:, Num_com - 1], ddof=1) ** 2)
        return tsquared, T2_lim, ellipse_radius

    def scaler(self, X_new):

        Cx = self.x_scaling[0, :]
        Sx = self.x_scaling[1, :]
        X_new = (X_new - Cx) / Sx

        return X_new

    def unscaler(self, X_new):
        Cx = self.x_scaling[0, :]
        Sx = self.x_scaling[1, :]
        X_new = (X_new * Sx) + Cx
        return X_new

    def MissEstimator(
            self, incom_data: np.ndarray = None):
        """
        It receives the incomplete data with nan in its missed values,


        Args:
            incom_data (np.ndarray, optional): data with missing values. Defaults to None.

        Returns:
            np.ndarray: data with estimated values for its missing elements
        """
        Estimated_block = incom_data.copy()

        # find columns with nan value
        incomplete_rows = np.where(np.isnan(Estimated_block).any(axis=1))[0]
        rows_with_all_nan = np.where(np.isnan(Estimated_block).all(axis=1))[0]
        Estimated_block[rows_with_all_nan] = 0
        incomplete_rows = incomplete_rows[incomplete_rows != rows_with_all_nan] if np.size(
            rows_with_all_nan) > 0 else incomplete_rows
        for i in incomplete_rows:
            x_new = incom_data[i, :].reshape(1, incom_data.shape[1])
            available_col = np.where(~np.isnan(x_new).any(axis=0))[0]
            no_avable_col = np.where(np.isnan(x_new).any(axis=0))[0]
            # scaling  x_new
            C_scaling = self.x_scaling[0, available_col]
            S_scaling = self.x_scaling[1, available_col]
            X_new_scaled = (x_new[0, available_col] - C_scaling) / S_scaling.reshape(
                1, -1
            )

            P_new = self.P[available_col, :]
            t_new = (X_new_scaled @ P_new) @ np.linalg.inv(P_new.T @ P_new)

            x_hat = t_new @ self.P.T
            Estimated_block[i, no_avable_col] = self.unscaler(x_hat).reshape(-1)[
                no_avable_col]
        return Estimated_block

    def visual_plot(
        self,
        score_axis=None,
        X_test=None,
        color_code_data=None,
        data_labeling=False,
        testing_labeling=False,
    ) -> None:
        # inner Functions

        def confidenceline(r1, r2, center):  # -> tuple:# -> tuple:
            t = np.linspace(
                0, 2 * np.pi, 100
            )  # Increase the number of points for a smoother ellipse
            x = center[0] + r1 * np.cos(t)
            y = center[1] + r2 * np.sin(t)
            return (x, y)

        def inner_ploter(
            y_data, position, legend_str, X_test=None, y_data_add=None, lim_line=None
        ) -> None:
            X_data = np.arange(1, len(y_data) + 1)
            legend_str1 = legend_str + " (Calibration Dataset)"
            legend_str2 = legend_str + "(New Dataset)"
            legend_str3 = legend_str + r"$_{lim}$"
            plt.subplot(2, 1, position[0])
            plt.plot(X_data, y_data, "bo", label=legend_str1)
            if X_test is not None:
                y_data = np.concatenate((y_data, y_data_add))
                X_data = np.arange(1, len(y_data) + 1)
                plt.plot(X_data[Num_obs:], y_data[Num_obs:],
                         "r*", label=legend_str2)
            plt.plot([1, X_data[-1] + 1], [lim_line]
                     * 2, "k--", label=legend_str3)
            plt.legend()
            plt.xlabel("Observations")
            plt.ylabel(legend_str)

        # Ploting Parameters
        Num_obs, Num_com = self.T.shape
        if score_axis is None:
            score_axis = np.array([1, min(2, Num_com)])

        # Create subplots
        fig1 = plt.figure(1)
        fig2 = plt.figure(2)

        # score plot
        tscore_x = self.T[:, score_axis[0] - 1]
        tscore_y = self.T[:, score_axis[1] - 1]

        r1 = self.ellipse_radius[score_axis[0] - 1]
        r2 = self.ellipse_radius[score_axis[1] - 1]
        xr, yr = confidenceline(r1, r2, np.array([0, 0]))
        label_str = f"Confidence Limit ({self.alpha * 100}%)"

        plt.figure(fig1.number)
        plt.suptitle("PCA Model Visual Plotting(scores)")
        plt.subplot(2, 2, (1, 2))
        plt.plot(xr, yr, "k--", label=label_str)
        if color_code_data is None:
            plt.scatter(tscore_x, tscore_y, color='b', marker='o',
                        s=50, label='Score (Training Dataset)')
        else:
            cmap = plt.get_cmap("viridis")
            norm = plt.Normalize(vmin=min(color_code_data),
                                 vmax=max(color_code_data))
            plt.scatter(
                tscore_x,
                tscore_y,
                c=color_code_data,
                cmap="viridis",
                s=100,
                label="Scores(Training Dataset)",
            )
            plt.colorbar()

        if data_labeling:
            for i in range(Num_obs):
                plt.text(
                    tscore_x[i],
                    tscore_y[i],
                    str(i + 1),
                    fontsize=10,
                    ha="center",
                    va="bottom",
                )

        # Testing Data
        tscore_testing, hoteling_t2_testing, spe_x_testing = None, None, None
        if X_test is not None:
            Num_new = X_test.shape[0]
            eval: pcaeval = self.evaluation(xtest=X_test)
            tscore_testing, hoteling_t2_testing, spe_x_testing = eval.tscore, eval.HT2, eval.spe

            t_score_x_new = tscore_testing[:, score_axis[0] - 1]
            t_score_y_new = tscore_testing[:, score_axis[1] - 1]
            plt.plot(t_score_x_new, t_score_y_new,
                     "r*", label="Score(New Data)")
            if testing_labeling:
                for i in range(Num_new):
                    plt.text(
                        [t_score_x_new[i]],
                        [t_score_y_new[i]],
                        str(i + 1),
                        color="red",
                        fontsize=10,
                        ha="center",
                        va="bottom",
                    )
        plt.legend()
        plt.xlabel(r"T$_{" + str(score_axis[0]) + r"}$ score")
        plt.ylabel(r"T$_{" + str(score_axis[1]) + r"}$ score")
        plt.title("PCA Score Plot Distribution")
        # Loading bar plots
        for k in range(2):
            Num_var_X = self.Xtrain_normal.shape[1]
            x_data = np.empty(Num_var_X, dtype=object)
            y_data = self.P[:, k]
            for j in range(Num_var_X):
                x_data[j] = "variable " + str(j + 1)
            plt.subplot(2, 2, k + 3)
            plt.bar(x_data, y_data, label="Loding" +
                    str(score_axis[k]), color="blue")
            plt.title("Loading of" + str(score_axis[k]) + "Component")
        plt.pause(0.1)
        plt.show(block=False)

        plt.figure(fig2.number)
        plt.suptitle("PCA Model Visual Plotting(Statistics)")
        # SPE_X Plot
        y_data = self.SPE_x[:, -1]
        lim_lin = self.SPE_lim_x[-1]
        inner_ploter(y_data, [1], r"SPE$_{X}$", X_test, spe_x_testing, lim_lin)
        # Hoteling T^2 Plot
        y_data = self.tsquared[:, -1]
        lim_lin = self.T2_lim[-1]
        inner_ploter(
            y_data, [
                2], r"HotelingT$^{2}$", X_test, hoteling_t2_testing, lim_lin
        )

        # Update layout for font sizes and other customization
        plt.pause(0.1)
        plt.show(block=False)
