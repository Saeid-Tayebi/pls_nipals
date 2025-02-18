import numpy as np
import pytest
from pls_nipals.pls import PlsClass as pls, plseval
from tests.refrence_model.pls_sklear import plsSkleanr

Num_observation = 30
xvar = 4
Noutput = 2
Num_testing = 1
n_component = Noutput+1             # Number of PLS components (=Number of X Variables)

# Calibration Dataset
X = np.random.rand(Num_observation, xvar)
Beta = np.random.rand(xvar, Noutput) * 2 - 1  # np.array([3,2,1])
Y = (X @ Beta)

# Targeted Output (For which Null space is to be explored)
X_test = np.random.rand(Num_testing, xvar)
Y_test = (X_test @ Beta)

# Center scalling X and Y
Cx, Sx = X.mean(axis=0), X.std(axis=0, ddof=1)
Cy, Sy = Y.mean(axis=0), Y.std(axis=0, ddof=1)

X, X_test = (X-Cx)/Sx, (X_test-Cx)/Sx
Y, Y_test = (Y-Cy)/Sy, (Y_test-Cy)/Sy


benchmarkpls = plsSkleanr(X, Y, n_component)
yfit_benchmark, Ttesb, T2tesb, SPEtesb, = benchmarkpls.eval(X_test)

mypls = pls().fit(X, Y, n_component=n_component)
eval: plseval = mypls.evaluation(X_test)
yfit, Ttes, T2tes, SPEtes = eval.yfit, eval.tscore, eval.HT2, eval.spex


def test_xhat():
    mypls2 = pls().fit(X, Y, n_component=xvar)
    x_hat: plseval = mypls2.evaluation(X).xhat
    assert np.allclose(x_hat, X)


def test_num_com_setter():
    mypls1 = pls().fit(X[1:5, :], Y[1:5, :], n_component=6)
    assert mypls1.n_component == 3

    mypls2 = pls().fit(X, Y, n_component=xvar+1)
    assert mypls2.n_component == xvar


def test_P():
    assert np.allclose(abs(benchmarkpls.P), abs(mypls.P), atol=.01)


def test_xtes():

    Threshold: float = .01
    assert np.allclose(abs(yfit), abs(yfit_benchmark), atol=Threshold)
    assert np.allclose(abs(Ttes), abs(Ttesb), atol=Threshold)
    assert np.allclose(abs(T2tes), abs(T2tesb), atol=Threshold)
    assert np.allclose(abs(SPEtes), abs(SPEtesb), atol=Threshold)


def test_x_predict():
    xpre = mypls.x_predict(Y_test, method=1)
    y_pre: plseval = mypls.evaluation(xtes=xpre).yfit
    assert np.allclose(y_pre, Y_test, atol=1e-5)


def test_null_space():
    """_summary_
    """
    Y_des = Y_test[0, :].reshape(1, -1)
    # all
    NS_t, NS_X, NS_Y = mypls.null_space_all(Y_des=Y_des)
    Y_pre = mypls.predict(NS_X)
    assert np.allclose(Y_pre.shape, NS_Y.shape)
    assert np.allclose(Y_pre, NS_Y)
    assert np.allclose(Y_pre, Y_des)
    assert np.allclose(NS_Y, Y_des)
    assert np.allclose(NS_t@mypls.Q.T, NS_Y)

    # single t to y
    for i in range(Y_des.shape[1]):
        NS_t, NS_X, NS_Y = mypls.null_space_single_col_t_to_Y(
            Y_des=Y_des, which_col=i)
        Y_pre = mypls.predict(NS_X)
        assert np.allclose(Y_pre.shape, NS_Y.shape)
        assert np.allclose(Y_pre[:, i], NS_Y[:, i])
        assert np.allclose(Y_pre[:, i], Y_des[:, i])
        assert np.allclose(NS_Y[:, i], Y_des[:, i])
        assert np.allclose((NS_t@mypls.Q.T)[:, i], NS_Y[:, i])

    # single X to y
    for i in range(Y_des.shape[1]):
        NS_t, NS_X, NS_Y = mypls.null_space_single_col_X_to_Y(
            Y_des=Y_des, which_col=i)
        Y_pre = mypls.predict(NS_X)
        assert np.allclose(Y_pre.shape, NS_Y.shape)
        assert np.allclose(Y_pre[:, i], NS_Y[:, i])
        assert np.allclose(Y_pre[:, i], Y_des[:, i])
        assert np.allclose(NS_Y[:, i], Y_des[:, i])
        assert np.allclose((NS_t@mypls.Q.T)[:, i], NS_Y[:, i])


def test_remove_zero_rows():
    """Test if rows with all zeros in X or Y are removed correctly before fitting."""

    # Case 1: X has rows of all zeros
    X_case1 = np.array([[0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 5, 6], [
                       7, 8, 9, 10, 11, 12], [11, 21, 31, 41, 51, 16]])
    Y_case1 = np.array([[1, 1, 1], [21, 31, 41], [15, 16, 17], [17, 18, 19]])
    pls_model = pls().fit(X_case1, Y_case1, n_component=2)
    assert pls_model.Xtrain_normal.shape[0] == 3
    assert pls_model.Ytrain_normal.shape[0] == 3
    # Case 2: Y has rows of all zeros
    X_case2 = np.array(
        [[1, 2, 3, 4, 5, 6], [17, 81, 19, 19, 11, 12], [13, 14, 15, 16, 17, 18]])
    Y_case2 = np.array([[0, 0, 0], [12, 31, 14], [51, 16, 17]])
    pls_model = pls().fit(X_case2, Y_case2, n_component=2)
    assert pls_model.Xtrain_normal.shape[0] == 2
    assert pls_model.Ytrain_normal.shape[0] == 2
    # Case 3: X and Y both have the same rows with all zeros
    X_case3 = np.array(
        [[0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 5, 6], [10, 18, 19, 41, 52, 32], [17, 81, 91, 11, 12, 52]])
    Y_case3 = np.array([[2, 3, 4], [0, 0, 0], [51, 16, 17], [22, 23, 24]])
    pls_model = pls().fit(X_case3, Y_case3, n_component=2)
    assert pls_model.Xtrain_normal.shape[0] == 2
    assert pls_model.Ytrain_normal.shape[0] == 2
    # Case 4: No zero rows (control case)
    X_case4 = np.array([[1, 2, 3, 4, 5, 6], [71, 18, 19, 12, 13, 32], [51, 58, 15, 52, 53, 55]])
    Y_case4 = np.array([[2, 3, 4], [152, 162, 127], [425, 246, 247]])
    pls_model = pls().fit(X_case4, Y_case4, n_component=2)
    assert pls_model.Xtrain_normal.shape[0] == 3
    assert pls_model.Ytrain_normal.shape[0] == 3

    # Case 5: X col with no var
    X_case1 = np.array([[1, 21, 31, 41],
                        [1, 15, 6, 17],
                        [1, 8, 19, 10]])  # First column has zero variance
    Y_case1 = np.array([[11, 12],
                        [13, 41],
                        [15, 16]])
    pls_model = pls().fit(X_case1, Y_case1, n_component=2)
    assert pls_model.Xtrain_normal.shape[1] == 3
    assert pls_model.Ytrain_normal.shape[1] == 2

    # Case 6: Y has a column with zero variance (all values are the same)
    X_case2 = np.array([[1, 21, 31],
                        [4, 5, 6],
                        [17, 18, 19]])
    Y_case2 = np.array([[1, 0],
                        [13, 0],
                        [51, 0]])  # Second column has zero variance
    pls_model = pls().fit(X_case2, Y_case2, n_component=2)
    assert pls_model.Xtrain_normal.shape[1] == 3
    assert pls_model.Ytrain_normal.shape[1] == 1

    # Case 7: Y has only one column, and one of its rows is zero (expect not to be removed)
    X_case3 = np.array([[11, 21, 31],
                        [4, 5, 6],
                        [7, 8, 9]])
    Y_case3 = np.array([[0],
                        [3],
                        [5]])  # The first row is zero but should NOT be removed
    pls_model = pls().fit(X_case3, Y_case3, n_component=2)
    assert pls_model.Xtrain_normal.shape[0] == 3
    assert pls_model.Ytrain_normal.shape[0] == 3

    # case study 8: no variance at all
    with pytest.raises(ValueError, match="data does not have any variance"):
        X_case3 = np.array([[1, 1, 1],
                            [2, 2, 2],
                            [3, 3, 3]])
        Y_case3 = np.array([[0],
                            [3],
                            [5]])
        pls_model = pls().fit(X_case3, Y_case3, n_component=2)
