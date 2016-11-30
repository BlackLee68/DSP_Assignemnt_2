clear all;
close all;
addpath('ARMAX_GARCH_K_Toolbox');
% Reset random stream for reproducibility.
rng(0,'twister');
JAssets = 10; % desired number of assets
% Generate means of returns between -0.02 and 0.05.
a = -0.01; b = 0.05;
mean_return = a + (b-a).*rand(JAssets,1);
% Generate standard deviations of returns between 0.008 and 0.006.
a = 0.08; b = 0.06;
stdDev_return = a + (b-a).*rand(JAssets,1);
%%
Ntime=200;
%% X: Each row is a time-instant. Each Column is an asset.
X=zeros(Ntime,JAssets);
for j=1:JAssets
    X(:,j)=mean_return(j)+stdDev_return(j)*randn(Ntime,1);
end
%% 
%% MF: mean forecast
%% VF: forecast of variance
MFpred=zeros(JAssets,1);
VFpred=zeros(JAssets);
for j=1:JAssets
    %% For each of the variables, fit the ARMA(1,1)-GARCH(1,1) model
    [parameters, stderrors, LLF, ht, resids, summary] = garch1(X(:,j),'GARCH', 'GAUSSIAN',1,1,0,1,1,0,[]);
    %% 1-step ahead Prediction of the mean and covariance of return 
    [MFpred(j), VFpred(j,j), ~, ~] = garchfor2(X(:,j), resids, ht, parameters, 'GARCH', 'GAUSSIAN',1,1,1,1,1);
end


a=ones(JAssets,1)'*(VFpred\ones(JAssets,1));
b=ones(JAssets,1)'*(VFpred\MFpred);
c=MFpred'*(VFpred\MFpred);

target=[-.1:5e-3:.1]';
risk=zeros(length(target),1);
for j=1:length(target)
   delta=a*c-b^2;
    lambda1=(c-b*target(j))/delta;
    lambda2=(a*target(j)-b)/delta;
    w=VFpred\(lambda1*ones(JAssets,1)+lambda2*MFpred);
    risk(j)=w'*VFpred*w;
end
plot(risk,target);
ylabel('Expected return')
xlabel('Variance')

%% Write your own code;  
hold on;
x0=(risk(32)+risk(33))/2;
y0=0.0575;
plot(x0,y0,'*');
hold on;
x1=risk(32);
y1=0.055;
x2=risk(33);
y2=0.06;
k1=(y2-y1)/(x2-x1);
b1=y0-k1*x0;
ff=@(risk) k1*risk+b1;
yy=ff(risk);
plot(risk,yy);
hold on;
x3=(risk(37)+risk(38))/2;
y3=0.0825;
plot(x3,y3,'*');
hold on;
x4=risk(37);
y4=0.08;
x5=risk(38);
y5=0.085;
k2=(y5-y4)/(x5-x4);
b2=y3-k2*x3;
ff=@(risk) k2*risk+b2;
yy=ff(risk);
plot(risk,yy);


