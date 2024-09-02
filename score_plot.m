function []=score_plot(model,score_axis,data_labeling_tr,new_data,data_labeling_new)
    %%% receive pls with the required scores that need to be plotted
    figure
    Num_obs=size(model.T,1);
    subplot(3,2,[1,2])
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

    


    plot(tx,ty,'bo',LineWidth=2)
    xlabel(['T _{' num2str(x_axis_score) '} score'])
    ylabel(['T_{ ' num2str(y_axis_score) '} score'])
    title('Score Plot Distribution')
     legend(['confidence limit (',num2str(model.alpha*100), '%)'],'Training Samples')
    hold on

    if data_labeling_tr        
            for i=1:Num_obs         
                text(tx(i), ty(i), num2str(i),'HorizontalAlignment','center',VerticalAlignment='baseline',FontSize=13)
            end
    end

    hoteling_T2_new=[];
    spe_x_new=[];
    spe_y_new=[];
    if nargin>3
        [~,T_score_new,hoteling_T2_new,spe_x_new,spe_y_new]=pls_evaluation(model,new_data);
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
%% spex
 subplot(3,2,3)
    y_data=[model.SPE_x(:,end); spe_x_new];
    x_data=1:length(y_data);
    y_lim=model.SPE_lim_x(end);
    title_str='SPE_{X}';
    inner_plotter(x_data,y_data,y_lim,Num_obs,title_str)

%%spey
subplot(3,2,4)
    y_data=[model.SPE_y(:,end); spe_y_new];
    x_data=1:length(y_data);
    y_lim=model.SPE_lim_y(end);
    title_str='SPE_{Y}';
    inner_plotter(x_data,y_data,y_lim,Num_obs,title_str)

%% hoteling T2
subplot(3,2,[5,6])
    y_data=[model.tsquared(:,end) ;hoteling_T2_new];
    x_data=1:length(y_data);
    y_lim=model.T2_lim(end);
    title_str='Hoteling T^{2}';
    inner_plotter(x_data,y_data,y_lim,Num_obs,title_str)

    function [x,y]=ellisedata(r1,r2,Center)

        %%% receive r1 and r2 and provude corresponding x and y data to plot a oval
        %%% it should be x^2/r1^2+y^2/r2^2=1                
                 t = linspace(0,2*pi);
                 x = Center(1)+r1*cos(t);
                 y = Center(2)+r2*sin(t);
    end
end

function []=inner_plotter(x_data,y_data,y_lim,Num_obs,title_str)
    plot(x_data(1:Num_obs),y_data(1:Num_obs),'bo',LineWidth=2)
    hold on
    plot([1 length(y_data)],y_lim*[1 1],'--k',LineWidth=2)
    if length(y_data)>Num_obs
        plot(x_data(Num_obs+1:end),y_data(Num_obs+1:end),'r*',LineWidth=2)
    end
    xlabel('Observations')
    title(title_str)
end