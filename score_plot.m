function []=score_plot(model,a,b)

    %%% receive pls or pca model with the required scores that we want to
    %%% plot and plot every thing with the ellips around them

        T=model.T;
        Num_obs=size(T,1);
        t1=T(:,a);
        t2=T(:,b);
        r1=model.ellipse_radius(a); 
        r2=model.ellipse_radius(b);
        
        plot(t1,t2,'mh',LineWidth=1)
        xlabel(['T _{' num2str(a) '} score'])
        ylabel(['T_{ ' num2str(b) '} score'])
        hold on
%         for i=1:Num_obs         
%             text(t1(i), t2(i), num2str(i),'HorizontalAlignment','center',VerticalAlignment='baseline',FontSize=13)
%         end
        
        Center=[0 0];
        [x,y]=ellisedata(r1,r2,Center);

        plot(x,y,'LineStyle','--','Color','black');

        set(gca, 'LineWidth', 2, 'FontSize', 15);
        

       

    function [x,y]=ellisedata(r1,r2,Center)

        %%% receive r1 and r2 and provude corresponding x and y data to plot a oval
        %%% it should be x^2/r1^2+y^2/r2^2=1                
                 t = linspace(0,2*pi);
                 x = Center(1)+r1*cos(t);
                 y = Center(2)+r2*sin(t);
    end
end