#%%
import numpy as np
from scipy.stats import chi2, f
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class PLS_structure:
    def __init__(self):
        # Initialize attributes if needed
        PLS_structure.T = None
        PLS_structure.S = None
        PLS_structure.P = None  
        PLS_structure.u =None
        PLS_structure.U =None
        PLS_structure.Q =None
        PLS_structure.Wstar =None
        PLS_structure.B_pls =None
        PLS_structure.x_hat_scaled = None
        PLS_structure.y_fit_scaled = None  
        PLS_structure.tsquared =None
        PLS_structure.T2_lim=None
        PLS_structure.ellipse_radius =None
        PLS_structure.SPE_x =None
        PLS_structure.SPE_lim_x =None
        PLS_structure.SPE_y =None
        PLS_structure.SPE_lim_y =None
        PLS_structure.Rsquared =None
        PLS_structure.covered_var =None
        PLS_structure.x_scaling =None
        PLS_structure.y_scaling =None
        PLS_structure.Xtrain_normal =None
        PLS_structure.Ytrain_normal =None
        PLS_structure.Xtrain_scaled =None
        PLS_structure.Ytrain_scaled =None
        PLS_structure.alpha =None
        PLS_structure.Null_Space =None
        PLS_structure.Num_com =None

def pls_nipals(X, Y, Num_com, alpha=0.95, to_be_scaled=1):
    
    if not bool(Num_com): 
        Num_com=X.shape[1]

    # Data Preparation
    X_orining = X
    Y_orining = Y
    Cx = np.mean(X, axis=0)
    Cy = np.mean(Y, axis=0)
    Sx = np.std(X, axis=0,ddof=1) + 1e-16
    Sy = np.std(Y, axis=0,ddof=1) + 1e-16

    if to_be_scaled==1:
        X = (X - Cx) / Sx
        Y = (Y - Cy) / Sy

    
    Num_obs = X.shape[0]
    K = X.shape[1]  # Num of X Variables
    M = Y.shape[1]  # Num of Y Variables
    X_0 = X
    Y_0 = Y

    # Blocks initialization
    W = np.zeros((K, Num_com))
    U = np.zeros((Num_obs, Num_com))
    Q = np.zeros((M, Num_com))
    T = np.zeros((Num_obs, Num_com))
    P = np.zeros_like(W)
    SPE_x = np.zeros_like(T)
    SPE_y = np.zeros_like(T)
    SPE_lim_x = np.zeros(Num_com)
    SPE_lim_y = np.zeros(Num_com)
    tsquared = np.zeros_like(T)
    T2_lim = np.zeros(Num_com)
    ellipse_radius = np.zeros(Num_com)
    Rx = np.zeros(Num_com)
    Ry = np.zeros(Num_com)

    # NIPALS Algorithm
    for i in range(Num_com):
        u = Y[:, np.argmax(np.var(Y_orining, axis=0,ddof=1))]
        while True:
            w = X.T @ u / (u.T @ u)
            w = w / np.linalg.norm(w)
            t1 = X @ w / (w.T @ w)
            q1 = Y.T @ t1 / (t1.T @ t1)
            unew = Y @ q1 / (q1.T @ q1)
            Error_x = np.sum((unew - u) ** 2)
            u = unew
            if Error_x < 1e-16:
                break

        P1 = X.T @ t1 / (t1.T @ t1)
        X = X - t1[:, None] @ P1[None, :]
        Y = Y - t1[:, None] @ q1[None, :]
        W[:, i] = w
        P[:, i] = P1
        T[:, i] = t1
        U[:, i] = unew
        Q[:, i] = q1
        # SPE_X
        SPE_x[:, i], SPE_lim_x[i], Rx[i] = SPE_calculation(T, P, X_0, alpha)

        # SPE_Y
        SPE_y[:, i], SPE_lim_y[i], Ry[i] = SPE_calculation(T, Q, Y_0, alpha)

        # Hotelling T2 Related Calculations
        tsquared[:, i], T2_lim[i], ellipse_radius[i] = T2_calculations(T[:, :i+1], i+1, Num_obs, alpha)

    Wstar = W @ np.linalg.pinv(P.T @ W)
    B_pls = Wstar @ Q.T
    S = np.linalg.svd(T.T @ T)[1]**0.5
    u = T / S

    # Null space
    A = Num_com
    KK = Y_orining.shape[1]
    if KK > A:
        Null_Space = 0
    elif KK == A:
        Null_Space = 1
    else:
        Null_Space = 2

    # Function Output
    mypls = PLS_structure()

    mypls.T=T
    mypls.S=S
    mypls.u=u
    mypls.P=P
    mypls.U=U
    mypls.Q=Q
    mypls.Wstar=Wstar
    mypls.B_pls=B_pls
    mypls.x_hat_scaled=T @ P.T

    mypls.y_fit_scaled=T @ Q.T,
    mypls.tsquared=tsquared
    mypls.T2_lim=T2_lim
    mypls.ellipse_radius=ellipse_radius
    mypls.SPE_x=SPE_x
    mypls.SPE_lim_x=SPE_lim_x
    mypls.SPE_y=SPE_y
    mypls.SPE_lim_y=SPE_lim_y
    mypls.Rsquared=np.array([Rx.T,Ry.T])*100
    mypls.covered_var=np.var(T, axis=0,ddof=1)
    mypls.x_scaling=np.vstack((Cx, Sx))
    mypls.y_scaling=np.vstack((Cy, Sy))
    mypls.Xtrain_normal=X_orining
    mypls.Ytrain_normal=Y_orining
    mypls.Xtrain_scaled=X_0
    mypls.Ytrain_scaled=Y_0
    mypls.alpha=alpha
    mypls.Null_Space=Null_Space
    mypls.Num_com=Num_com
    

    return mypls

def pls_evaluation(pls_model:PLS_structure,X_new):
  """
  receive pls model and new observation and calculate its
   y_pre,T_score,Hotelin_T2,SPE_X,SPE_Y
  """
  #if X_new.ndim==1:
  #      X_new=X_new.reshape(1,X_new.size)  
  y_pre,T_score=Y_fit_Calculation(pls_model,X_new)
  X_new_scaled,Y_new_scaled=scaler(pls_model,X_new,y_pre)
  
  Hotelin_T2=np.sum((T_score/np.std(pls_model.T,axis=0,ddof=1))**2,axis=1)
  SPE_X,_,_ = SPE_calculation(T_score, pls_model.P, X_new_scaled, pls_model.alpha)
  SPE_Y,_,_ = SPE_calculation(T_score, pls_model.Q, Y_new_scaled, pls_model.alpha)

  return y_pre,T_score,Hotelin_T2,SPE_X,SPE_Y


def SPE_calculation(score, loading, Original_block, alpha):
    # Calculation of SPE and limits
    X_hat = score @ loading.T
    Error = Original_block - X_hat
    #Error.reshape(-1,loading.shape[1])
    spe = np.sum(Error**2, axis=1)
    m = np.mean(spe)
    v = np.var(spe,ddof=1)
    spe_lim = v / (2 * m) * chi2.ppf(alpha, 2 * m**2 / v)
    Rsquare = 1 - np.var(Error,ddof=1) / np.var(Original_block,ddof=1) # not applicaple for pls vali
    return spe, spe_lim, Rsquare

def T2_calculations(T, Num_com, Num_obs, alpha):
    # Calculation of Hotelling T2 statistics
    tsquared = np.sum((T / np.std(T, axis=0,ddof=1))**2, axis=1)
    T2_lim = (Num_com * (Num_obs**2 - 1)) / (Num_obs * (Num_obs - Num_com)) * f.ppf(alpha, Num_com, Num_obs - Num_com)
    ellipse_radius = np.sqrt(T2_lim * np.std(T[:, Num_com - 1],ddof=1)**2)
    return tsquared, T2_lim, ellipse_radius

def Y_fit_Calculation(pls_model:PLS_structure, X_new):
    x_new_scaled,_ = scaler(pls_model,X_new,0)
    y_fit_scaled = x_new_scaled @ pls_model.B_pls
    T_score=x_new_scaled @ pls_model.Wstar
    _,y_fit = unscaler(pls_model,0,y_fit_scaled)
    return y_fit,T_score

def scaler(pls_model:PLS_structure,X_new,Y_new):

    Cx=pls_model.x_scaling[0,:]
    Sx=pls_model.x_scaling[1,:]
    X_new=(X_new-Cx)/Sx
    #if not Y_new==0:
    Cy=pls_model.y_scaling[0,:]
    Sy=pls_model.y_scaling[1,:]
    Y_new=(Y_new-Cy)/Sy
    return X_new,Y_new
    
def unscaler(pls_model:PLS_structure,X_new,Y_new):
    Cx=pls_model.x_scaling[0,:]
    Sx=pls_model.x_scaling[1,:]
    X_new=(X_new * Sx) + Cx
    #if not Y_new==0:
    Cy=pls_model.y_scaling[0,:]
    Sy=pls_model.y_scaling[1,:]
    Y_new=(Y_new * Sy) + Cy
    return X_new,Y_new


def visual_plot(pls_model, score_axis=None, X_test=None, data_labeling=False, testing_labeling=False):

    # inner Functions
    def confidenceline(r1, r2, center):
        t = np.linspace(0, 2 * np.pi, 100)  # Increase the number of points for a smoother ellipse
        x = center[0] + r1 * np.cos(t)
        y = center[1] + r2 * np.sin(t)
        return x, y
    
    def inner_ploter(y_data,position,legend_str,X_test=None,y_data_add=None,legend_str2=None):       
        X_data = np.arange(1, len(y_data) + 1)
        fig.add_trace(go.Scatter(x=X_data, y=y_data, mode='markers', marker=dict(color='blue', size=10), name=legend_str,showlegend=True),
                  row=position[0], col=position[1])
        if X_test is not None:
            y_data = np.concatenate((y_data, y_data_add))
            X_data = np.arange(1, len(y_data) + 1)
            fig.add_trace(go.Scatter(x=X_data[Num_obs:], y=y_data[Num_obs:], mode='markers', marker=dict(color='red', symbol='star', size=12), name=legend_str2,showlegend=True),
                      row=position[0], col=position[1])
        fig.add_trace(go.Scatter(x=[1, X_data[-1] + 1], y=[pls_model.T2_lim[-1]] * 2, mode='lines', line=dict(color='black', dash='dash'), name='Hoteling T^2 Lim',showlegend=False),
                  row=position[0], col=position[1])
        fig.update_xaxes(
        tickmode='linear',  # Ensures all ticks are shown linearly
        tick0=2,            # Starting tick (adjust if needed)
        dtick=1,            # Interval between ticks
        range=[0.5, len(X_data)+0.5],
        row=position[0], col=position[1] ) # Apply only to the specific subplot

    # Ploting Parameters
    Num_obs, Num_com = pls_model.T.shape
    if score_axis is None:
        score_axis = np.array([1, min(2, Num_com)])

    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'PLS Score Plot Distribution','SPE_X', 'SPE_Y','Hoteling T^2 Plot'),
        specs=[[{"colspan": 2}, None],   # Row 1: Plot 1 spans columns 1 and 2
               [{}, {}],                        # Row 2: Normal 2-column layout
               [{"colspan":2}, {}]]             # Row 3: Normal 2-column layout
       ,row_heights=[0.5, 0.25, 0.25],
    )
    # axis labeling
    fig.update_xaxes(title_text='T '+str(score_axis[0])+'score',row=1,col=1)
    fig.update_yaxes(title_text='T '+str(score_axis[1])+'score',row=1,col=1)
    fig.update_xaxes(title_text='Observations',row=3,col=1)
    #score plot
    tscore_x = pls_model.T[:, score_axis[0] - 1]
    tscore_y = pls_model.T[:, score_axis[1] - 1]

    r1 = pls_model.ellipse_radius[score_axis[0] - 1]
    r2 = pls_model.ellipse_radius[score_axis[1] - 1]
    xr, yr = confidenceline(r1, r2, np.array([0, 0]))
    label_str = f'Confidence Limit ({pls_model.alpha * 100}%)'

    fig.add_trace(go.Scatter(x=xr, y=yr, mode='lines', line=dict(color='black', dash='dash'), name=label_str,showlegend=True),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=tscore_x, y=tscore_y, mode='markers', marker=dict(color='blue', size=10), name='Score(Training Dataset)',showlegend=True),
                  row=1, col=1)
    
    
    if data_labeling:
        for i in range(Num_obs):
            fig.add_trace(go.Scatter(x=[tscore_x[i]], y=[tscore_y[i]], text=str(i + 1), mode='text', textposition='top center',showlegend=False),
                          row=1, col=1)
    # Testing Data
    tscore_testing, hoteling_t2_testing, spe_x_testing, spe_y_testing=None,None,None,None
    if X_test is not None:
        Num_new = X_test.shape[0]
        _, tscore_testing, hoteling_t2_testing, spe_x_testing, spe_y_testing = pls_evaluation(pls_model, X_test)

        t_score_x_new = tscore_testing[:, score_axis[0] - 1]
        t_score_y_new = tscore_testing[:, score_axis[1] - 1]

        fig.add_trace(go.Scatter(x=t_score_x_new, y=t_score_y_new, mode='markers', marker=dict(color='red', symbol='star', size=12), name='Score(New Data)',showlegend=True),
                      row=1, col=1)
        if testing_labeling:
            for i in range(Num_new):
                fig.add_trace(go.Scatter(x=[t_score_x_new[i]], y=[t_score_y_new[i]], text=str(i + 1), mode='text', textposition='top center',showlegend=False),
                              row=1, col=1)

    # SPE_X Plot
    y_data = pls_model.SPE_x[:, -1]
    inner_ploter(y_data,[2,1],'SPE_X(Training Data)',X_test,spe_x_testing,'SPE_X(New Data)')
    # SPE_Y Plot
    y_data = pls_model.SPE_y[:, -1]
    inner_ploter(y_data,[2,2],'SPE_Y(Training Data)',X_test,spe_y_testing,'SPE_Y(New Data)')
    # Hoteling T^2 Plot
    y_data = pls_model.tsquared[:, -1]
    inner_ploter(y_data,[3,1],'Hoteling T2(Training Data)',X_test,hoteling_t2_testing,'Hoteling T2(New Data)')

  
    # Update layout for font sizes and other customization
    fig.update_layout(
    title_text='PLS Model Visual Plotting',
    title_x=0.5,
    font=dict(size=15),
    legend=dict(x=1, y=1, traceorder='normal'),
    showlegend=True,
    # Use annotations for X and Y labels for the entire figure
    )
    fig.show()

