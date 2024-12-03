#%%
import numpy as np
import pls_module as pls_m
from MyPlsClass import MyPls as pls_c

# Model further settings
np.set_printoptions(precision=4)

# Generating data
Num_observation=30
Ninput=5
Noutput=2
Num_testing=10
Num_com = 2             # Number of PLS components (=Number of X Variables)
alpha = 0.95            # Confidence limit (=0.95)
scores_plt=np.array([1,2])

X =np.random.rand(Num_observation,Ninput)
Beta=np.random.rand(Ninput,Noutput) * 2 -1 #np.array([3,2,1])
Y=(X @ Beta)

X_test=np.random.rand(Num_testing,Ninput)
Y_test=(X_test @ Beta)
scores_plt=np.array([1,2])
#%%
# Model implementation as a Module
pls_model=pls_m.pls_nipals(X,Y,Num_com,alpha)

# Show the validation for a new or testing observation
y_pre,T_score,Hotelin_T2,SPE_X,SPE_Y_pre=pls_m.pls_evaluation(pls_model,X_test)
print(f'Y_pre={y_pre}\n',f'T_score={T_score}\n',f'Hotelin_T2={Hotelin_T2}\n',f'SPE_X={SPE_X}\n',f'SPE_Y_pre={SPE_Y_pre}\n')

# Visually see the data distributions
pls_m.visual_plot(pls_model,scores_plt,X_test=X_test,data_labeling=True,color_code_data=Y[:,0]) # 

#%% 
# Model implementation as a Class
MyPlsModel=pls_c()
MyPlsModel.train(X,Y,Num_com,alpha)
y_pre,T_score,Hotelin_T2,SPE_X,SPE_Y_pre=MyPlsModel.evaluation(X_test)
MyPlsModel.visual_plot(scores_plt,X_test,color_code_data=Y[:,0],data_labeling=True)


print(f'Y_pre={y_pre}\n',f'T_score={T_score}\n',f'Hotelin_T2={Hotelin_T2}\n',f'SPE_X={SPE_X}\n',f'SPE_Y_pre={SPE_Y_pre}\n')

#%% PLS MI checking
MyPlsModel=pls_c()
MyPlsModel.train(X,Y,alpha=alpha,Num_com=1)
x_des,y_pre_MI=MyPlsModel.MI(Y_des=Y_test[1,:].reshape(1,-1),method=1)
print('org MI',x_des,y_pre_MI,Y_test[1,:])
x_des,y_pre_MI=MyPlsModel.MI(Y_des=Y_test[1,:].reshape(1,-1),method=2)
print('Sug MI',x_des,y_pre_MI,Y_test[1,:])
#%% PLS Null Space checking

MyPlsModel=pls_c()
MyPlsModel.train(X,Y,Num_com=3,alpha=alpha)
NS_t,NS_X,NS_Y=MyPlsModel.NS_all(Y_des=Y[1,:].reshape(1,-1),MI_method=1)
MyPlsModel.visual_plot(X_test=NS_X)

NS_t,NS_X,NS_Y=MyPlsModel.NS_single(which_col=1,Num_point=1000,Y_des=Y[1,:].reshape(1,-1),MI_method=1)
MyPlsModel.visual_plot(X_test=NS_X)

NS_t,NS_X,NS_Y=MyPlsModel.NS_XtoY(which_col=2,Num_point=1000,Y_des=Y[1,:].reshape(1,-1),MI_method=1)
MyPlsModel.visual_plot(X_test=NS_X)
# %%
