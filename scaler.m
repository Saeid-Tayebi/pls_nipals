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