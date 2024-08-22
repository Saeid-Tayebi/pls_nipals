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