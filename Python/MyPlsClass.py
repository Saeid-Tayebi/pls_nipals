import numpy as np
from scipy.stats import chi2, f
import matplotlib.pyplot as plt
from MyPcaClass import MyPca as pca
from sklearn.linear_model import LinearRegression


class MyPls:
    def __init__(self):
        # Initialize attributes if needed
        self.T = None
        self.S = None
        self.P = None  
        self.u =None
        self.U =None
        self.Q =None
        self.Wstar =None
        self.B_pls =None
        self.x_hat_scaled = None
        self.y_fit_scaled = None  
        self.tsquared =None
        self.T2_lim=None
        self.ellipse_radius =None
        self.SPE_x =None
        self.SPE_lim_x =None
        self.SPE_y =None
        self.SPE_lim_y =None
        self.Rsquared =None
        self.covered_var =None
        self.x_scaling =None
        self.y_scaling =None
        self.Xtrain_normal =None
        self.Ytrain_normal =None
        self.Xtrain_scaled =None
        self.Ytrain_scaled =None
        self.alpha =None
        self.Null_Space =None
        self.Num_com =None

    def train(self,X, Y, Num_com=None, alpha=0.95, to_be_scaled=1):

        if Num_com is None: 
            Num_com=self.Num_components_sugg(X)
        if Num_com>(X.shape[0]-1):
            Num_com=X.shape[0]-1
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
            SPE_x[:, i], SPE_lim_x[i], Rx[i] = self.SPE_calculation(T, P, X_0, alpha,is_train=1)

            # SPE_Y
            SPE_y[:, i], SPE_lim_y[i], Ry[i] = self.SPE_calculation(T, Q, Y_0, alpha,is_train=1)

            # Hotelling T2 Related Calculations
            tsquared[:, i], T2_lim[i], ellipse_radius[i] = self.T2_calculations(T[:, :i+1], i+1, Num_obs, alpha)

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

        self.T=T
        self.S=S
        self.u=u
        self.P=P
        self.U=U
        self.Q=Q
        self.Wstar=Wstar
        self.B_pls=B_pls
        self.x_hat_scaled=T @ P.T

        self.y_fit_scaled=T @ Q.T,
        self.tsquared=tsquared
        self.T2_lim=T2_lim
        self.ellipse_radius=ellipse_radius
        self.SPE_x=SPE_x
        self.SPE_lim_x=SPE_lim_x
        self.SPE_y=SPE_y
        self.SPE_lim_y=SPE_lim_y
        self.Rsquared=np.array([Rx.T,Ry.T])*100
        self.covered_var=np.array([Rx, Ry]).T * 100
        self.x_scaling=np.vstack((Cx, Sx))
        self.y_scaling=np.vstack((Cy, Sy))
        self.Xtrain_normal=X_orining
        self.Ytrain_normal=Y_orining
        self.Xtrain_scaled=X_0
        self.Ytrain_scaled=Y_0
        self.alpha=alpha
        self.Null_Space=Null_Space
        self.Num_com=Num_com

        return self
    def Num_components_sugg(self,Z,ploting=False):
        '''
        Z can be either X or Y
        determines the number of components that describe the data quite good using the eigenvalue_greater_than_one_rule
        '''
        pca_model=pca()
        pca_model.train(Z)
        eig_val=pca_model.covered_var
        Num_com_sugg=np.sum(eig_val>1)
        if ploting==True:
            plt.figure()
            plt.bar(range(1,eig_val.shape[1]+1),eig_val.reshape(-1),label='Covered Variance')
            plt.xlabel('Components')
            plt.ylabel('Variance Covered')
            plt.plot([0,eig_val.shape[1]+1],[1,1],'k--',label='Threshold Line')
            plt.legend()
            plt.show()
        # plot the eig_Val using barchart to show if suggested A is rational or not

        return Num_com_sugg
    def evaluation(self,X_new):
        """
        receive pls model and new observation and calculate its
        y_pre,T_score,Hotelin_T2,SPE_X,SPE_Y
        """
        #if X_new.ndim==1:
        #      X_new=X_new.reshape(1,X_new.size)  
        y_pre,T_score=self.Y_fit_Calculation(X_new)
        X_new_scaled,Y_new_scaled=self.scaler(X_new,y_pre)
        
        Hotelin_T2=np.sum((T_score/np.std(self.T,axis=0,ddof=1))**2,axis=1)
        SPE_X,_,_ = self.SPE_calculation(T_score, self.P, X_new_scaled, self.alpha)
        SPE_Y,_,_ = self.SPE_calculation(T_score, self.Q, Y_new_scaled, self.alpha)

        return y_pre,T_score,Hotelin_T2,SPE_X,SPE_Y


    def SPE_calculation(self,score, loading, Original_block, alpha,is_train=0):
        # Calculation of SPE and limits
        X_hat = score @ loading.T
        Error = Original_block - X_hat
        #Error.reshape(-1,loading.shape[1])
        spe = np.sum(Error**2, axis=1)
        spe_lim, Rsquare=None,None
        if is_train==1:
            m = np.mean(spe)
            v = np.var(spe,ddof=1)
            spe_lim = v / (2 * m) * chi2.ppf(alpha, 2 * m**2 / (v+1e-15))
            Rsquare = 1 - np.var(Error,ddof=1) / np.var(Original_block,ddof=1) 
        return spe, spe_lim, Rsquare

    def T2_calculations(self,T, Num_com, Num_obs, alpha):
        # Calculation of Hotelling T2 statistics
        tsquared = np.sum((T / np.std(T, axis=0,ddof=1))**2, axis=1)
        T2_lim = (Num_com * (Num_obs**2 - 1)) / (Num_obs * (Num_obs - Num_com)) * f.ppf(alpha, Num_com, Num_obs - Num_com)
        ellipse_radius = np.sqrt(T2_lim * np.std(T[:, Num_com - 1],ddof=1)**2)
        return tsquared, T2_lim, ellipse_radius

    def Y_fit_Calculation(self, X_new):
        x_new_scaled,_ = self.scaler(X_new=X_new)
        y_fit_scaled = x_new_scaled @ self.B_pls
        T_score=x_new_scaled @ self.Wstar
        _,y_fit = self.unscaler(Y_new=y_fit_scaled)
        return y_fit,T_score
    def MI(self,Y_des,method=1):
        '''
        This method receives Y_des and calculate its corresponding X_new using the general PLS Model Inversion solution
        if mode is 2 then it uses my suggested MI
        '''
        Wstar=self.Wstar
        Q=self.Q
        P=self.P
        T=self.T
        NS=self.Null_Space
        __,Y_des_scled=self.scaler(Y_new=Y_des)

        if method==1:
            if NS==0:   # not direct answer
                t_des=np.linalg.inv(Q.T @ Q) @ Q.T @ Y_des_scled.T
            elif NS==1: # only One answer
                t_des=np.linalg.inv(Q) @ Y_des_scled.T
            elif NS==2: #NS exist
                t_des= Q.T @ np.linalg.inv(Q @ Q.T) @ Y_des_scled.T
            t_des=t_des.T
        elif method==2: # My suggested MI
            Y_pca=self.Ytrain_normal
            T_pls=self.T
            pca_num_com=np.min((Y_pca.shape[1],T_pls.shape[1]))
            pca_model=pca()
            pca_model.train(Y_pca,pca_num_com)
            T_pca=pca_model.T
            __,t_des_pca,__,__=pca_model.evaluation(Y_des)
            regression_model = LinearRegression()
            regression_model.fit(T_pca, T_pls)
            t_des=regression_model.predict(t_des_pca)
            
        x_des_scaled= t_des @ P.T
        x_des,__=self.unscaler(X_new=x_des_scaled)
        y_pre,__=self.Y_fit_Calculation(x_des)
        return x_des,y_pre    
        
    def scaler(self,X_new=None,Y_new=None):
        X_new_scaled,Y_new_scaled=None,None
        if X_new is not None:
            Cx=self.x_scaling[0,:]
            Sx=self.x_scaling[1,:]
            X_new_scaled=(X_new-Cx)/Sx
        if Y_new is not None:
            Cy=self.y_scaling[0,:]
            Sy=self.y_scaling[1,:]
            Y_new_scaled=(Y_new-Cy)/Sy
        return X_new_scaled,Y_new_scaled
    
    def unscaler(self,X_new=None,Y_new=None):
        X_new_unscaled,Y_new_unscaled=None,None
        if X_new is not None:
            Cx=self.x_scaling[0,:]
            Sx=self.x_scaling[1,:]
            X_new_unscaled=(X_new * Sx) + Cx
        if Y_new is not None:
            Cy=self.y_scaling[0,:]
            Sy=self.y_scaling[1,:]
            Y_new_unscaled=(Y_new * Sy) + Cy
        return X_new_unscaled,Y_new_unscaled

    def NS_all(self, Num_point=100, Y_des=None, MI_method=1):
        ''' Calculate NS for all columns at the same time using SVD based on Macgregor paper '''
        from scipy.linalg import sqrtm
        if self.Null_Space!=2:
            print('Null Space Does Not Exist')
            return None
        
        X_ds,__=self.MI(Y_des,MI_method)   # direct x answer
        __,t_ds,__,__,__=self.evaluation(X_ds)

        T=self.T
        P=self.P
        Q=self.Q
        S=sqrtm(T.T @ T)
        

        left_singular, v, Vt = np.linalg.svd(S@Q.T, full_matrices=True)
     
        i=len(v)
        G=left_singular[:,i:]
        NS_dim=G.shape[1]

        Ub= np.min(self.ellipse_radius)
        Lb=-Ub

        gamma= (Ub-Lb)*np.random.rand(10*Num_point,NS_dim) + Lb

        # refine the acceptable gamma
        u_ds = t_ds @ np.linalg.inv(S)
        x_preScaled= (u_ds+gamma@G.T)@S@P.T
        x_pre,__=self.unscaler(X_new=x_preScaled,Y_new=None)
        Ypre,t_score,HotelingT2,__,__=self.evaluation(x_pre)
        isvalid=HotelingT2<self.T2_lim[-1]
        
        NS_t=t_score[isvalid,:]
        NS_X=x_pre[isvalid,:]
        NS_Y=Ypre[isvalid,:]

        return NS_t,NS_X,NS_Y


    def NS_single(self, which_col,Num_point=100, Y_des=None, MI_method=1):
        ''' Calculate NS for each column separately once at a time based on Garcia paper '''
        def meshgrid_multi_dimension(Lb,Ub,NumRow,NumCol):
            outData=np.zeros([NumRow,NumCol])
            randomData=np.linspace(Lb,Ub,10*NumRow)
            for i in range(NumCol):
                outData[:,i]=np.random.choice(randomData, size=NumRow, replace=True)
            return outData
        
        X_ds,__=self.MI(Y_des,MI_method)   # direct x answer
        __,t_ds,__,__,__=self.evaluation(X_ds)
        P=self.P
        Q=self.Q
        NumNs=Q.shape[1]
        Ub= np.min(self.ellipse_radius)
        Lb=-Ub
        NumRow=Num_point
        NumCol=NumNs-1

        if which_col>Q.shape[1]:
            print('Null Space Does Not Exist')
            return None

        delta_t=meshgrid_multi_dimension(Lb,Ub,NumRow,NumCol)
        which_col=which_col-1
        c=np.zeros([1,NumCol])
        for j in range(NumCol):
            c[0,j]= -Q[which_col,j]/Q[which_col,-1]
            
        z=np.sum(c*delta_t,axis=1)
        delta_t = np.column_stack((delta_t, z))
        ptNS=t_ds+delta_t
        x_preScaled= ptNS @ P.T
        x_pre,__=self.unscaler(X_new=x_preScaled,Y_new=None)
        Ypre,t_score,HotelingT2,__,__=self.evaluation(x_pre)
        isvalid=HotelingT2<self.T2_lim[-1]
        NS_t=t_score[isvalid,:]
        NS_X=x_pre[isvalid,:]
        NS_Y=Ypre[isvalid,:]

        return NS_t,NS_X,NS_Y
    
    def NS_XtoY(self, which_col,Num_point=100, Y_des=None, MI_method=1):
        ''' Calculate NS for each column of Y seperately directly using the X data  '''
        def meshgrid_multi_dimension(Lb,Ub,NumRow,NumCol):
            outData=np.zeros([NumRow,NumCol])
            for i in range(NumCol):
                randomData=np.linspace(Lb[i],Ub[i],10*NumRow)
                outData[:,i]=np.random.choice(randomData, size=NumRow, replace=True)
            return outData
        
        X_ds,__=self.MI(Y_des,MI_method)   # direct x answer
        X_ds_scaled,__=self.scaler(X_new=X_ds)
        __,t_ds,__,__,__=self.evaluation(X_ds)

        B=self.B_pls
        NumNs=B.shape[0]
        Ub= np.min(self.Xtrain_scaled,axis=0)
        Lb=-Ub
        NumRow=Num_point
        NumCol=NumNs-1

        if which_col>B.shape[1]:
            print('Null Space Does Not Exist')
            return None

        delta_x=meshgrid_multi_dimension(Lb,Ub,NumRow,NumCol)
        which_col=which_col-1
        c=np.zeros([1,NumCol])
        for j in range(NumCol):
            c[0,j]= -B[j,which_col]/B[-1,which_col]
            
        z=np.sum(c*delta_x,axis=1)
        delta_x = np.column_stack((delta_x, z))
        pxNS=X_ds_scaled+delta_x
        x_preScaled=pxNS
        x_pre,__=self.unscaler(X_new=x_preScaled,Y_new=None)
        Ypre,t_score,HotelingT2,__,__=self.evaluation(x_pre)
        isvalid=HotelingT2<self.T2_lim[-1]
        NS_t=t_score[isvalid,:]
        NS_X=x_pre[isvalid,:]
        NS_Y=Ypre[isvalid,:]

        return NS_t,NS_X,NS_Y

    def visual_plot(self, score_axis=None, X_test=None,color_code_data=None, data_labeling=False, testing_labeling=False):
        def confidenceline(r1, r2, center):
                t = np.linspace(0, 2 * np.pi, 100)  # Increase the number of points for a smoother ellipse
                x = center[0] + r1 * np.cos(t)
                y = center[1] + r2 * np.sin(t)
                return x, y
        def inner_ploter(y_data,position,legend_str,X_test=None,y_data_add=None,lim_line=None):       
            X_data = np.arange(1, len(y_data) + 1)
            legend_str1=legend_str+' (Calibration Dataset)'
            legend_str2=legend_str+'(New Dataset)'
            legend_str3=legend_str+r'$_{lim}$'
            plt.subplot(3,2,position[0])
            plt.plot(X_data,y_data,'bo',label=legend_str1)
            if X_test is not None:
                y_data = np.concatenate((y_data, y_data_add))
                X_data = np.arange(1, len(y_data) + 1)
                plt.plot(X_data[Num_obs:],y_data[Num_obs:],'r*',label=legend_str2)
            plt.plot([1, X_data[-1] + 1],[lim_line] * 2,'k--',label=legend_str3)   
            plt.legend()
            plt.xlabel('Observations')
            plt.ylabel(legend_str)

        # Ploting Parameters
        Num_obs, Num_com = self.T.shape
        if score_axis is None:
            score_axis = np.array([1, min(2, Num_com)])

        # Create subplots
        plt.subplot(3,2,(1,2))
        plt.suptitle('PLS Model Visual Plotting(scores)')
        # axis labeling
        plt.xlabel('T '+str(score_axis[0])+'score')
        plt.ylabel('T '+str(score_axis[1])+'score')

        #score plot
        tscore_x = self.T[:, score_axis[0] - 1]
        tscore_y = self.T[:, score_axis[1] - 1]

        r1 = self.ellipse_radius[score_axis[0] - 1]
        r2 = self.ellipse_radius[score_axis[1] - 1]
        xr, yr = confidenceline(r1, r2, np.array([0, 0]))
        label_str = f'Confidence Limit ({self.alpha * 100}%)'
        plt.plot(xr,yr,'k--',label=label_str)
        if color_code_data is None:
            plt.scatter(tscore_x, tscore_y, color='b', marker='o', s=50, label='Score (Training Dataset)')
        else:
            cmap = plt.get_cmap('viridis')
            norm = plt.Normalize(vmin=min(color_code_data), vmax=max(color_code_data))  
            plt.scatter(tscore_x,tscore_y,c=color_code_data, cmap='viridis',s=100,label='Scores(Training Dataset)')
            plt.colorbar() 
        
        if data_labeling:
            for i in range(Num_obs):
                plt.text(tscore_x[i],tscore_y[i],str(i+1),fontsize=10, ha='center', va='bottom')
        # Testing Data
        tscore_testing, hoteling_t2_testing, spe_x_testing, spe_y_testing=None,None,None,None
        if X_test is not None:
            Num_new = X_test.shape[0]
            _, tscore_testing, hoteling_t2_testing, spe_x_testing, spe_y_testing = self.evaluation(X_test)

            t_score_x_new = tscore_testing[:, score_axis[0] - 1]
            t_score_y_new = tscore_testing[:, score_axis[1] - 1]
            plt.scatter(t_score_x_new, t_score_y_new, color='r', marker='*', s=70, label='Score(New Dataset)')

            if data_labeling:
                for i in range(Num_new):
                    plt.text(t_score_x_new[i],t_score_y_new[i],str(i+1),fontsize=10, ha='center', va='bottom',color='red')

        
        
        # SPE_X Plot
        y_data = self.SPE_x[:, -1]
        lim_line=self.SPE_lim_x[-1]
        inner_ploter(y_data,[3],'SPE_X',X_test,spe_x_testing,lim_line)
        # SPE_Y Plot
        y_data = self.SPE_y[:, -1]  
        lim_line=self.SPE_lim_y[-1]
        inner_ploter(y_data,[4],'SPE_Y',X_test,spe_y_testing,lim_line)
        
        # Hoteling T^2 Plot
        y_data = self.tsquared[:, -1]
        lim_line=self.T2_lim[-1]
        inner_ploter(y_data,[(5,6)],'Hoteling T2',X_test,hoteling_t2_testing,lim_line)
        plt.legend(loc='best')
        plt.pause(0.1)
        plt.show(block=False)