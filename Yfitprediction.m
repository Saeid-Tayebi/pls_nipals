 function [y_hat,t_score]=Yfitprediction(pls_model,x_pre)

        %%% Receive the right PLS model and X_new( unscaled and un
        %%% centered) calculate the corresponding Y-fit and unscale it
        [x_pre,~]=scaler(pls_model,x_pre,0);
        t_score=x_pre*pls_model.Wstar;
        y_hat=x_pre*pls_model.B_pls;        
        [~,y_hat]=Unscaler(pls_model,0,y_hat);
    end