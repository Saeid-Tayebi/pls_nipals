clear all
clc
close all


    load generated_data.mat
    X=generated_data(:,1:3);
    Y=generated_data(:,4:end);
Num_com=3;
alfa=0.95;
    mypls=pls_nipals(X,Y,Num_com,alfa)