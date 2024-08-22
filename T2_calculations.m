function [tsquared, T2_lim,ellipse_radius]=T2_calculations(T,Num_com,Num_obs,alfa)

%%% recieve Score Matrix, the current applied number of components,num of
%%% observations and alfa and return all Hotelling T2 related calculations
%%% including tsquared, T2_lim and ellipse_radius
            tsquared=sum((T./std(T)).^2,2);
            T2_lim=(Num_com*(Num_obs^2-1))/(Num_obs*(Num_obs-Num_com))*finv(alfa,Num_com,(Num_obs-Num_com));
            ellipse_radius=(T2_lim*std(T(:,Num_com))^2)^0.5;

end