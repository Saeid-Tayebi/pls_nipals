function [y_pre,T_score,Hoteling_T2,SPE_x,SPE_y]=pls_evaluation(pls_model,x_new)
%%% receive pls model and new observation and calculate its score, T2,SPEx
%%% and SPEy

            [y_pre,~]=Yfitprediction(pls_model,x_new);
            [Obs_new_x,Obs_new_y]=scaler(pls_model,x_new,y_pre);
            alpha=pls_model.alpha;


            %Function output
            
            T_score=Obs_new_x*pls_model.Wstar;

            Hoteling_T2=sum((T_score./std(pls_model.T)).^2,2);

            [SPE_x,~,~]=SPE_calculation(T_score, pls_model.P,Obs_new_x,alpha);

            [SPE_y,~,~]=SPE_calculation(T_score, pls_model.Q,Obs_new_y,alpha);
           
end

 function [y_hat,t_score]=Yfitprediction(pls_model,x_pre)
        %%% Receive the PLS model and X_new( unscaled and un
        %%% centered) calculate the corresponding Y-fit and unscale it
        [x_pre,~]=scaler(pls_model,x_pre,0);
        t_score=x_pre*pls_model.Wstar;
        y_hat=x_pre*pls_model.B_pls;        
        [~,y_hat]=Unscaler(pls_model,0,y_hat);
 end

 function [x_scaled,y_scaled]=scaler(pls_model,x,y)
%%% receive pls and x and y and scaled them based on the model used

        Cx=pls_model.x_scaling(1,:);
        Sx=pls_model.x_scaling(2,:);
        x_scaled=(x-Cx)./Sx;
if ~isempty(y)
        Cy=pls_model.y_scaling(1,:);
        Sy=pls_model.y_scaling(2,:);    
        y_scaled=(y-Cy)./Sy;
else
    y_scaled=[];
end
        
 end

 function [x,y]=Unscaler(pls_model,x_scaled,y_scaled)
%%% receive pls and x_scaled and y_scaled and unscaled them based on the model used

        y=y_scaled;

        Cx=pls_model.x_scaling(1,:);
        Sx=pls_model.x_scaling(2,:);
        x=(x_scaled.*Sx)+Cx;

        if ~isempty (y_scaled)
                Cy=pls_model.y_scaling(1,:);
                Sy=pls_model.y_scaling(2,:);    
                y=(y_scaled.*Sy)+Cy;
        end       
 end

 function [spe,spe_lim,Rsquare]=SPE_calculation(score, loading,Original_block,alpha)
%%% receive score,loading, original block (scaled format) and alpha, and calculate the Error
%%% and SPE and the SPE_lim as well as Rsquared

            X_hat=score*loading';
            Error=Original_block-X_hat;
            spe=sum(Error.*Error,2);
            m=mean(spe);
            v=var(spe);
            spe_lim=v/(2*m)*chi2inv(alpha,2*m^2/v);

            %Rsquared
            Rsquare=(1-var(Error)/var(Original_block));
             
end
