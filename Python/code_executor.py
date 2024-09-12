#%%
import numpy as np
import pls_module as pls_m
from MyPlsClass import MyPls as pls_c

# Model further settings
np.set_printoptions(precision=4)

# Generating data
Num_observation=30
Number_variables=3
X =np.random.rand(Num_observation,Number_variables)
Beta=np.array([3,2,1])
Y=(X@Beta.T).reshape(-1,1)

# Set parameters
Num_com = 2             # Number of PLS components (=Number of X Variables)
alpha = 0.95            # Confidence limit (=0.95)
X_test=np.array([[0.9,0.1,0.2],[0.5 , 0.4 , 0.9]])
scores_plt=np.array([1,2])
#%%
# Model implementation as a Module
pls_model=pls_m.pls_nipals(X,Y,Num_com,alpha)

# Show the validation for a new or testing observation
y_pre,T_score,Hotelin_T2,SPE_X,SPE_Y_pre=pls_m.pls_evaluation(pls_model,X_test)
print(f'Y_pre={y_pre}\n',f'T_score={T_score}\n',f'Hotelin_T2={Hotelin_T2}\n',f'SPE_X={SPE_X}\n',f'SPE_Y_pre={SPE_Y_pre}\n')

# Visually see the data distributions
pls_m.visual_plot(pls_model,scores_plt,X_test,True,True) # 

#%% 
# Model implementation as a Class
MyPlsModel=pls_c()
MyPlsModel.train(X,Y,Num_com,alpha)
y_pre,T_score,Hotelin_T2,SPE_X,SPE_Y_pre=MyPlsModel.evaluation(X_test)
MyPlsModel.visual_plot(scores_plt)

print(f'Y_pre={y_pre}\n',f'T_score={T_score}\n',f'Hotelin_T2={Hotelin_T2}\n',f'SPE_X={SPE_X}\n',f'SPE_Y_pre={SPE_Y_pre}\n')

# %%
