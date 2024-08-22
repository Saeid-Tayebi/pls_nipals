function mypls=pls_nipals(X,Y,Num_com,alfa,Mc)
%%% this function only receive X, and Y (Not centered, Not scaled) and the numer of required
%%% components and report T,P,Wstar,Q,U,B_pls, tsquared (if means for both),
%%% T2_lim, SPE_lim_x, SPE_lim_y, SPE_x,spey, centering and scaling factors 

%% Further Setting 
        numInputs = nargin;
        if numInputs>4
        to_be_scaled=Mc.tobescaled;
        else
        to_be_scaled=1;
        end
        if Num_com==0
        Num_com=Num_Com_determination(X);
        end


%%
        X_orining=X;
        Y_orining=Y;
        Cx=mean(X);Cy=mean(Y);
        Sx=std(X)+1e-16;Sy=std(Y)+1e-16;
        X=(X-Cx)./(Sx);
        Y=(Y-Cy)./(Sy);
        Num_obs=size(X,1);
        K=size(X,2); %Num of X Variables
        M=size(Y,2); %Num of Y Variables
        X_0=X;
        Y_0=Y;

        % Blocks initialization

        W=zeros(K,Num_com);
        U=zeros(Num_obs,Num_com);
        Q=zeros(M,Num_com);
        T=zeros(Num_obs,Num_com);
        P=zeros(size(W));
        SPE_x=zeros(size(T));
        SPE_y=zeros(size(T));
        SPE_lim_x=zeros(1,Num_com);
        SPE_lim_y=zeros(1,Num_com);
        tsquared=zeros(size(T));
        T2_lim=zeros(1,Num_com);
        ellipse_radius=zeros(1,Num_com);
        Rx=zeros(1,Num_com);
        Ry=zeros(1,Num_com);

        %% NIPALS Algorithm

        for i=1:Num_com
            [~,b]=max(var(Y_orining));
            u=Y(:,b);
            while true           
                w=(X'*u)/(u'*u);
                w=w/norm(w);
                t1=(X*w)/(w'*w);
                q1=(Y'*t1)/(t1'*t1);
                unew=(Y*q1)/(q1'*q1);             
                Error_x=unew-u;
                Error_x=Error_x'*Error_x;
                u=unew;
                if Error_x<1e-15
                    break
                end
            end
            P1=(X'*t1)/(t1'*t1);
            E=X-(t1*P1');
            F=Y-(t1*q1'); 
            X=E;
            Y=F;        
            W(:,i)=w;
            P(:,i)=P1;
            T(:,i)=t1;
            U(:,i)=unew;
            Q(:,i)=q1;
            % SPE_X
            [SPE_x(:,i),SPE_lim_x(i),Rx(i)]=SPE_calculation(T, P,X_0,alfa);
           
            % SPE_Y
            [SPE_y(:,i),SPE_lim_y(i),Ry(i)]=SPE_calculation(T, Q,Y_0,alfa);

            % Hotelling T2 Related Calculations
            [tsquared(:,i), T2_lim(i),ellipse_radius(i)]=T2_calculations(T(:,1:i),i,Num_obs,alfa);
            
         
        end
            Wstar=W/(P'*W);
            B_pls=Wstar*Q';
            S=(T'*T)^0.5;
            u=T/S;

        %% Addition Parameters 
            % Null space
        A=Num_com;
        KK=size(Y_orining,2);
          if  KK>A
              Null_Space=0;
          elseif KK==A
              Null_Space=1;
          elseif KK<A
               Null_Space=2;
          end
            
        %% Function Output 
        mypls.T=T;
        mypls.S=S;
        mypls.u=u;
        mypls.P=P;
        mypls.U=U;
        mypls.Q=Q;
        mypls.Wstar=Wstar;
        mypls.B_pls=B_pls;
        mypls.x_hat_scaled=T*P';
        mypls.y_fit_scaled=T*Q';
        mypls.x_hat_Unscaled=((T*P').*Sx)+Cx;
        mypls.y_fit_Unscaled=((T*Q').*Sy)+Cy;
        mypls.tsquared=tsquared;
        mypls.T2_lim=T2_lim;
        mypls.ellipse_radius=ellipse_radius; % differs from ProMV but match with bano2017 and facco2015
        mypls.SPE_x=SPE_x;
        mypls.SPE_lim_x=SPE_lim_x;
        mypls.SPE_y=SPE_y;
        mypls.SPE_lim_y=SPE_lim_y;
        mypls.Rsquared=[Rx' Ry']*100;
        mypls.covered_var=var(T);
        mypls.x_scaling=[Cx;Sx];
        mypls.y_scaling=[Cy;Sy];
        mypls.Xtrain_normal=X_orining;
        mypls.Ytrain_normal=Y_orining;
        mypls.Xtrain_scaled=X_0;
        mypls.Ytrain_scaled=Y_0;
        mypls.alfa=alfa;
        mypls.Null_Space=Null_Space;
        mypls.Num_com=Num_com;


end

function [spe,spe_lim,Rsquare]=SPE_calculation(score, loading,Original_block,alfa)

%%% receive score,loading, original block (scaled format) and alfa, and calculate the Error
%%% and SPE and the SPE_lim as well as Rsquared

            X_hat=score*loading';
            Error=Original_block-X_hat;
            spe=sum(Error.*Error,2);
            m=mean(spe);
            v=var(spe);
            spe_lim=v/(2*m)*chi2inv(alfa,2*m^2/v);

            %Rsquared
            Rsquare=(1-var(Error)/var(Original_block));
             
end

function [tsquared, T2_lim,ellipse_radius]=T2_calculations(T,Num_com,Num_obs,alfa)

%%% recieve Score Matrix, the current applied number of components,num of
%%% observations and alfa and return all Hotelling T2 related calculations
%%% including tsquared, T2_lim and ellipse_radius
            tsquared=sum((T./std(T)).^2,2);
            T2_lim=(Num_com*(Num_obs^2-1))/(Num_obs*(Num_obs-Num_com))*finv(alfa,Num_com,(Num_obs-Num_com));
            ellipse_radius=(T2_lim*std(T(:,Num_com))^2)^0.5;

end