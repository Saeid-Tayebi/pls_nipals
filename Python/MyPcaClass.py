import numpy as np
from scipy.stats import chi2, f
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class MyPca:
    def __init__(self):
        # Initialize attributes if needed
        self.T = None
        self.P = None  
        self.x_hat =None 
        self.tsquared =None
        self.T2_lim=None
        self.ellipse_radius =None
        self.SPE_x =None
        self.SPE_lim_x =None
        self.Rsquared =None
        self.covered_var =None
        self.x_scaling =None
        self.Xtrain_normal =None
        self.Xtrain_scaled =None
        self.alpha =None
        self.Num_com =None

    def train(self,X, Num_com=None, alpha=0.95, to_be_scaled=1):
        
        if Num_com is None: 
            Num_com=self.Num_components_sugg(X)
        if Num_com>(X.shape[0]-1):
            Num_com=X.shape[0]-1
        # Data Preparation
        X_orining = X
        Cx = np.mean(X, axis=0)
        Sx = np.std(X, axis=0,ddof=1) + 1e-16

        if to_be_scaled==1:
            X = (X - Cx) / Sx
        
        Num_obs = X.shape[0]
        K = X.shape[1]  # Num of X Variables
        X_0 = X

        # Blocks initialization
        T = np.zeros((Num_obs, Num_com))
        P = np.zeros((K,Num_com))
        covered_var=np.zeros((1,Num_com))
        SPE_x = np.zeros_like(T)
        SPE_lim_x = np.zeros(Num_com)
        tsquared = np.zeros_like(T)
        T2_lim = np.zeros(Num_com)
        ellipse_radius = np.zeros(Num_com)
        Rx = np.zeros(Num_com)

        # NIPALS Algorithm
        for i in range(Num_com):
            t1 = X[:, np.argmax(np.var(X_orining, axis=0,ddof=1))]
            while True:
                P1=(t1.T@X)/(t1.T@t1)
                P1=P1/np.linalg.norm(P1)
                t_new=((P1@X.T)/(P1.T@P1)).T

                Error = np.sum((t_new - t1) ** 2)
                t1=t_new
                if Error < 1e-16:
                    break
            x_hat=t1.reshape(-1,1)@P1.reshape(1,-1)
            X=X-x_hat
            P[:,i]=P1
            T[:,i]=t1

            covered_var[:,i]=np.var(t1,axis=0,ddof=1)
            # SPE_X
            SPE_x[:, i], SPE_lim_x[i], Rx[i] = self.SPE_calculation(T, P, X_0, alpha,is_train=1)

            # Hotelling T2 Related Calculations
            tsquared[:, i], T2_lim[i], ellipse_radius[i] = self.T2_calculations(T[:, :i+1], i+1, Num_obs, alpha)

        # Function Output
        self.T=T
        self.P=P
        self.x_hat=((T @ P.T)*Sx)+Cx
        self.tsquared=tsquared
        self.T2_lim=T2_lim
        self.ellipse_radius=ellipse_radius
        self.SPE_x=SPE_x
        self.SPE_lim_x=SPE_lim_x
        self.Rsquared=Rx.T*100
        self.covered_var=covered_var
        self.x_scaling=np.vstack((Cx, Sx))
        self.Xtrain_normal=X_orining
        self.Xtrain_scaled=X_0
        self.alpha=alpha
        self.Num_com=Num_com
        
        return self
    def Num_components_sugg(self,Z,ploting=False):
        '''
        Z can be either X or Y
        determines the number of components that describe the data quite good using the eigenvalue_greater_than_one_rule
        '''
        self.train(Z,Num_com=Z.shape[1])
        eig_val=self.covered_var
        Num_com_sugg=np.sum(eig_val>1)
        if ploting==True:
            plt.figure()
            plt.bar(range(1,eig_val.shape[1]+1),eig_val.reshape(-1),label='Covered Variance')
            plt.xlabel('Components')
            plt.ylabel('Variance Covered')
            plt.plot([0,eig_val.shape[1]+1],[1,1],'k--',label='Threshold Line')
            plt.legend()
            plt.show()

        return Num_com_sugg
    def evaluation(self,X_new): 
        """
        receive pca model and new observation and calculate its
        x_hat,T_score,Hotelin_T2,SPE_X
        """
        x_new_scaled=self.scaler(X_new)

        T_score=x_new_scaled @ self.P
        x_hat_new_scaled=T_score @ self.P.T

        x_hat_new=self.unscaler(x_hat_new_scaled)
        Hotelin_T2=np.sum((T_score/np.std(self.T,axis=0,ddof=1))**2,axis=1)
        SPE_X,_,_ = self.SPE_calculation(T_score, self.P, x_new_scaled, self.alpha)

        return x_hat_new,T_score,Hotelin_T2,SPE_X

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
            Rsquare = 1 - np.var(Error,ddof=1) / np.var(Original_block,ddof=1) # not applicaple for pls vali
        return spe, spe_lim, Rsquare

    def T2_calculations(self,T, Num_com, Num_obs, alpha):
        # Calculation of Hotelling T2 statistics
        tsquared = np.sum((T / np.std(T, axis=0,ddof=1))**2, axis=1)
        T2_lim = (Num_com * (Num_obs**2 - 1)) / (Num_obs * (Num_obs - Num_com)) * f.ppf(alpha, Num_com, Num_obs - Num_com)
        ellipse_radius = np.sqrt(T2_lim * np.std(T[:, Num_com - 1],ddof=1)**2)
        return tsquared, T2_lim, ellipse_radius


    def scaler(self,X_new):

        Cx=self.x_scaling[0,:]
        Sx=self.x_scaling[1,:]
        X_new=(X_new-Cx)/Sx
        
        return X_new
    
    def unscaler(self,X_new):
        Cx=self.x_scaling[0,:]
        Sx=self.x_scaling[1,:]
        X_new=(X_new * Sx) + Cx
        return X_new
    
    def MissEstimator(self,incom_data:np.ndarray=None,comple_data:np.ndarray=None):
        '''
        It receives the incomplete data with None in its missed columns, if the actual value is also given then the 
        estimation accuracy will be given as well
        '''
        Estimated_block=np.zeros_like(incom_data)
        Estimation_quality=np.zeros((incom_data.shape[0],1))
        for i in range(incom_data.shape[0]):
           x_new=incom_data[i,:].reshape(1,incom_data.shape[1])
           available_col = np.where(~np.isnan(x_new).any(axis=0))[0]
           no_avable_col = np.where(np.isnan(x_new).any(axis=0))[0]
           # scaling  x_new
           C_scaling=self.x_scaling[0,available_col]
           S_scaling=self.x_scaling[1,available_col]
           X_new_scaled=(x_new[0,available_col]-C_scaling)/S_scaling.reshape(1,-1)
           
           P_new=self.P[available_col,:]
           t_new=(X_new_scaled @ P_new) @ np.linalg.inv(P_new.T @ P_new)
           
           x_hat=t_new @ self.P.T
           Estimated_block[i,:]=self.unscaler(x_hat).reshape(1,-1)
           if comple_data is not None:
               actual=comple_data[i,no_avable_col].reshape(1,-1)
               estimated=Estimated_block[i,no_avable_col].reshape(1,-1)
               Estimation_quality[i]=self.Single_obs_error_calculation(actual,estimated,self.Xtrain_normal[:,no_avable_col])
        return Estimated_block,Estimation_quality
    
    def Single_obs_error_calculation(self,y_act,y_pre,Y_act=None):
        '''
        it receives actual and predicted_value and calculte the single prediction accuracy
        it need Y (the entire Y block to make sure there is not bias caused by the magnitude of th ecolomns)
        '''
        if Y_act is None:
            Y_act=y_act
        pa=np.zeros_like(y_act)
        for i in range(y_act.shape[1]):
            base_value=np.min(Y_act[:,i])
            scaled_Y=Y_act[:,i]-base_value
            Y_avr=np.mean(scaled_Y)
            error=np.abs(y_act[:,i]-y_pre[:,i])
            pa[:,i]=1-(error/Y_avr)
        Prediction_accuracy=np.mean(pa,axis=1)
        return Prediction_accuracy 
    def visual_plot(self, score_axis=None, X_test=None, color_code_data=None, data_labeling=False, testing_labeling=False):
        # inner Functions
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
            plt.subplot(2,1,position[0])
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
        fig1=plt.figure(1)
        fig2=plt.figure(2)

        #score plot
        tscore_x = self.T[:, score_axis[0] - 1]
        tscore_y = self.T[:, score_axis[1] - 1]

        r1 = self.ellipse_radius[score_axis[0] - 1]
        r2 = self.ellipse_radius[score_axis[1] - 1]
        xr, yr = confidenceline(r1, r2, np.array([0, 0]))
        label_str = f'Confidence Limit ({self.alpha * 100}%)'

        plt.figure(fig1.number)
        plt.suptitle('PCA Model Visual Plotting(scores)')
        plt.subplot(2,2,(1,2))
        plt.plot(xr,yr,'k--',label=label_str)
        if color_code_data is None:
            plt.plot(tscore_x,tscore_y,'ob',s=10,label='Scores(Training Dataset)')
        else:
            cmap = plt.get_cmap('viridis')
            norm = plt.Normalize(vmin=min(color_code_data), vmax=max(color_code_data))  
            plt.scatter(tscore_x,tscore_y,c=color_code_data, cmap='viridis',s=100,label='Scores(Training Dataset)')
            plt.colorbar()
        
        if data_labeling:  
            for i in range(Num_obs):   
                plt.text(tscore_x[i],tscore_y[i],str(i+1),fontsize=10,ha='center',va='bottom')

        # Testing Data
        tscore_testing, hoteling_t2_testing, spe_x_testing=None,None,None
        if X_test is not None:
            Num_new = X_test.shape[0]
            _, tscore_testing, hoteling_t2_testing, spe_x_testing = self.evaluation(X_test)

            t_score_x_new = tscore_testing[:, score_axis[0] - 1]
            t_score_y_new = tscore_testing[:, score_axis[1] - 1]
            plt.plot(t_score_x_new,t_score_y_new,'r*',label='Score(New Data)')
            if testing_labeling:
                for i in range(Num_new):
                    plt.text([t_score_x_new[i]],[t_score_y_new[i]],str(i+1),color='red',fontsize=10,ha='center',va='bottom')
        plt.legend()
        plt.xlabel(r'T$_{'+str(score_axis[0])+r'}$ score')
        plt.ylabel(r'T$_{'+str(score_axis[1])+r'}$ score')
        plt.title('PCA Score Plot Distribution')
        # Loading bar plots
        for k in range(2):
            Num_var_X=self.Xtrain_normal.shape[1]
            x_data = np.empty(Num_var_X, dtype=object)
            y_data=self.P[:,k]
            for j in range(Num_var_X):
                x_data[j]='variable '+str(j+1)
            plt.subplot(2,2,k+3)
            plt.bar(x_data,y_data,label='Loding'+str(score_axis[k]),color='blue')
            plt.title('Loading of'+str(score_axis[k])+'Component')
        plt.pause(0.1)
        plt.show(block=False)

        plt.figure(fig2.number)
        plt.suptitle('PCA Model Visual Plotting(Statistics)')
        # SPE_X Plot
        y_data = self.SPE_x[:, -1]
        lim_lin=self.SPE_lim_x[-1]
        inner_ploter(y_data,[1],r'SPE$_{X}$',X_test,spe_x_testing,lim_lin)
        # Hoteling T^2 Plot
        y_data = self.tsquared[:, -1]
        lim_lin=self.T2_lim[-1]
        inner_ploter(y_data,[2],r'HotelingT$^{2}$',X_test,hoteling_t2_testing,lim_lin)    
    
        # Update layout for font sizes and other customization
        plt.pause(0.1)
        plt.show(block=False)

