function [y_pre,T_score,T2,SPE_x,SPE_y]=pls_evaluation(pls_model,x_new)
%%% receive pls model and new observation and calculate its score, T2,SPEx
%%% and SPEy

            [y_pre,~]=Yfitprediction(pls_model,x_new);
            [Obs_new_x,Obs_new_y]=scaler(pls_model,x_new,y_pre);
            alfa=pls_model.alfa;


            %Function output
            
            T_score=Obs_new_x*pls_model.Wstar;

            T2=sum((T_score./std(pls_model.T)).^2,2);

            [SPE_x,~,~]=SPE_calculation(T_score, pls_model.P,Obs_new_x,alfa);

            [SPE_y,~,~]=SPE_calculation(T_score, pls_model.Q,Obs_new_y,alfa);

           
            
end