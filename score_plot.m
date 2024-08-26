function []=score_plot(model,score_axis,data_labeling_tr,new_data,data_labeling_new)
    %%% receive pls with the required scores that need to be plotted
    figure

    if nargin<2
        score_axis=[1,size(model.T,2)];
    end

     if nargin<3
        data_labeling_tr=1;
    end

    x_axis_score=score_axis(1);
    y_axis_score=score_axis(2);

    r1=model.ellipse_radius(x_axis_score); 
    r2=model.ellipse_radius(y_axis_score);
    Center=[0 0];
    [x,y]=ellisedata(r1,r2,Center);
    plot(x,y,'LineStyle','--','Color','black');
    hold on
    set(gca, 'LineWidth', 2, 'FontSize', 15);

    T=model.T;    
    tx=T(:,x_axis_score);
    ty=T(:,y_axis_score);

    


    plot(tx,ty,'kh',LineWidth=1)
    xlabel(['T _{' num2str(x_axis_score) '} score'])
    ylabel(['T_{ ' num2str(y_axis_score) '} score'])
    title('Score Plot Distribution')
     legend(['confidence limit (',num2str(model.alpha*100), '%)'],'Training Samples')
    hold on

    if data_labeling_tr
         Num_obs=size(T,1);
            for i=1:Num_obs         
                text(tx(i), ty(i), num2str(i),'HorizontalAlignment','center',VerticalAlignment='baseline',FontSize=13)
            end
    end

if nargin>3
[~,T_score_new,~,~,~]=pls_evaluation(model,new_data);
    tx_new=T_score_new(:,x_axis_score);
    ty_new=T_score_new(:,y_axis_score);
    
    plot(tx_new,ty_new,'bh',LineWidth=2)
    legend(['confidence limit (',num2str(model.alpha*100), '%)'],'Training Samples', 'New Samples')
if nargin<5
    data_labeling_new=1;
end

if data_labeling_new==1
    
     for i=1:size(new_data,1)         
                text(tx_new(i), ty_new(i), num2str(i),'Color','red','HorizontalAlignment','center',VerticalAlignment='baseline',FontSize=13)
    end

end

end


            

    function [x,y]=ellisedata(r1,r2,Center)

        %%% receive r1 and r2 and provude corresponding x and y data to plot a oval
        %%% it should be x^2/r1^2+y^2/r2^2=1                
                 t = linspace(0,2*pi);
                 x = Center(1)+r1*cos(t);
                 y = Center(2)+r2*sin(t);
    end
end