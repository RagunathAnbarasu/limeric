clc
%Date: 19/11/2020
%Authors : Aviroop, Ragunath
%This function displays the message rate over time, the input parameters
%are alpha, beta, K and T

K   = [100, 200, 300, 200, 100, 400];  %Users varying over a time-period
T   = [100, 100, 100, 100, 100, 100];  %Time-periods for user variation

alpha   = 0.1;     	%Value of alpha
beta    = 1/max(K);	%Value of beta

rg  = 1200;	%Channel Busy Ratio (CBR) target (message/sec)

%Using fixed values of alpha and beta
r_fixed     = zeros; %The message rate for each individual user   
r_ini_fixed = zeros; %Define the initial rates when starting a new cycle
R_avg_fixed     = zeros(1,sum(T));  %Average message rate
R_total_fixed   = zeros(1,sum(T));  %Total message rate

%Using Optimization
r_opt       = zeros; %The message rate for each individual user 
r_ini_opt   = zeros; %Define the initial rates when starting a new cycle
R_avg_opt   = zeros(1,sum(T)); %Average message rate
R_total_opt   = zeros(1,sum(T));  %Total message rate

%Using Reinforcement Learning (Q-Learning)
r_RL     = zeros; %The message rate for each individual user   
r_ini_RL = zeros; %Define the initial rates when starting a new cycle
R_avg_RL = zeros(1,sum(T));  %Average message rate
R_total_RL   = zeros(1,sum(T));  %Total message rate

T_var   = 1;    %Start of time period

%Loop to determine the average rate over time
for i = 1:max(size(K))
    if eq(i,1)
        r_ini_fixed(1:K(1),1)   = 10*rand(K(1),1); %Set the initial values for K users
        r_ini_opt(1:K(1),1)     = 10*rand(K(1),1); %Set the initial values for K users
        r_ini_RL(1:K(1),1)      = 10*rand(K(1),1); %Set the initial values for K users
    end
    
    %Evaluating the fixed scenario
    r_fixed = rate_fixed(alpha,beta,T(i),K(i),r_ini_fixed,rg);    %Rate function calculation
    r_ini_fixed = r_fixed(1:K(i),T(i));     %Redefine the initial starting message rates
    R_avg_fixed(1,T_var:(T_var+T(i)-1))     = mean(r_fixed(:,1:T(i))); %Average message rate over a given interval
    R_total_fixed(1,T_var:(T_var+T(i)-1))	= sum(r_fixed(:,1:T(i)));  %Total message rate over a given interval

    %Evaluating the optimized scenario
    r_opt = rate_opt(T(i),K(i),r_ini_opt,rg);	%Rate function calculation
    r_ini_opt = r_opt(1:K(i),T(i));     %Redefine the initial starting message rates
    R_avg_opt(1,T_var:(T_var+T(i)-1))   = mean(r_opt(:,1:T(i))); %Average message rate over a given interval
    R_total_opt(1,T_var:(T_var+T(i)-1))	= sum(r_opt(:,1:T(i)));  %Total message rate over a given interval
   
    %Evaluating using RL scenario
    r_RL = rate_RL(T(i),K(i),r_ini_RL,rg);	%Rate function calculation
    r_ini_RL = r_RL(1:K(i),T(i));     %Redefine the initial starting message rates
    R_avg_RL(1,T_var:(T_var+T(i)-1))    = mean(r_RL(:,1:T(i))); %Average message rate over a given interval
    R_total_RL(1,T_var:(T_var+T(i)-1))  = sum(r_RL(:,1:T(i)));  %Total message rate over a given interval

    T_var = T_var + T(i);   %Alterations in time-period
end

figure(1)
plot(0:sum(T)-1,R_avg_fixed,'Linewidth',3.0)
hold on
plot(0:sum(T)-1,R_avg_opt,'Linewidth',3.0)
hold on
plot(0:sum(T)-1,R_avg_RL,'g','Linewidth',3.0)
legend('Fixed scenario','Optimized scenario','Reinforcement Learning scenario')
set(gca,'FontSize',28)
set(gcf,'color','w');
grid on
title('Average message rate over interval')
xlabel('Interval')
ylabel('Average message rate')

figure(2)
plot(0:sum(T)-1,R_total_fixed,'c','Linewidth',3.0)
hold on
plot(0:sum(T)-1,R_total_opt,'m','Linewidth',3.0)
hold on
plot(0:sum(T)-1,R_total_RL,'y','Linewidth',3.0)
legend('Fixed scenario','Optimized scenario','Reinforcement Learning scenario')
set(gca,'FontSize',28)
set(gcf,'color','w');
grid on
title('Total message rate over interval')
xlabel('Interval')
ylabel('Total message rate')


%%%Fixed scenario: Using fixed paramters of alpha and beta to ...
%calculate the message rate for each interval
function rate_calculation = rate_fixed(alpha,beta,T,K,r_initial,rg) 
r  = zeros(K,T);    %Initializing the message rates for K users

%Setting the initial message rate for existing users
K_var   = max(size(r_initial));

if gt(K,K_var)
    r(1:K_var,1)    = r_initial(1:K_var);
    r(K_var+1:K,1)  = 1*rand(K - K_var,1); %Added users are randmly initialized
else
    r(1:K,1) = r_initial(1:K);
end

rc(1:T) = zeros;    %The measured CBR
rc(1)   = 1100;     %Initlialize rc

%Setting the message rates for the users based on the formula from
%the references
for t = 2:T
    e   = rg - rc(t-1);
    r(:,t)      = (1 - alpha) * r(:,t-1) + beta * (e);
    rc(t)       = (1 - alpha - K * beta) * rc(t-1) + (K * beta * rg);
end

rate_calculation = r(:,1:T);
end


%%%Optimized scenario: Using paramters of alpha and beta using optimization...
%to calculate the message rate for each interval
function rate_calculation = rate_opt(T,K,r_initial,rg) 
r  = zeros(K,T);    %Initializing the message rates for K users

%Setting the initial message rate for existing users
K_var   = max(size(r_initial));

if gt(K,K_var)
    r(1:K_var,1)    = r_initial(1:K_var);
    r(K_var+1:K,1)  = 1*rand(K - K_var,1); %Added users are randmly initialized
else
    r(1:K,1) = r_initial(1:K);
end

r_var   = r; 
rc(1:T) = zeros;    %The measured CBR
rc(1)   = 1100;     %Initlialize rc

rc_opt_initial 	= rc(1);

%Initial values of alpha and beta

alpha   = 0.1;
beta    = 1/K;

%Calculating the optimal alpha and beta values
for t = 2:10
    e   = rg - rc(t-1);
    r_var(:,t)      = (1 - alpha) * r_var(:,t-1) + beta * (e);
    
    %Determine alpha and beta by solving the error function
    err     = @(v)(rg -1*((1 - v(1) - K*v(2))*rc(t-1) + K*v(2)*rg)); %error function
    A       = [0,1];    %Less than equal to
    b       = 1.1/K;    %Beta value based on stability criteria
    Aeq     = [];       %Equivalency
    beq     = [];       %Equivalency
    lb      = [0.1,1/1000]; %Lower bound
    ub      = [0.9,1.1/K];  %Upper bound
    v0      = [0.9,1/K];    %Starting values
    
    opt = optimset('Display','off');
    v   = fmincon(err,v0,A,b,Aeq,beq,lb,ub,[],opt); %Minimize the error function 
    
    alpha   = v(1); %Alpha value obtained
    beta    = v(2); %Beta value obtained
    rc(t)   = (1 - alpha - K * beta) * rc(t-1) + (K * beta * rg);   %Set the measured CBR
end

%Reinitializing the values of alpha and beta
alpha_opt     = alpha;
beta_opt      = beta;
rc(1:T) = zeros;    %The measured CBR
rc(1)   = rc_opt_initial;        

%Setting the message rates for the users based on the formula from
%the references
for t = 2:T
    e   = rg - rc(t-1);
    r(:,t) = (1 - alpha_opt) * r(:,t-1) + beta_opt * (e);
    rc(t)  = (1 - alpha_opt - K * beta_opt) * rc(t-1) + (K * beta_opt * rg);   %Set the measured CBR
end

rate_calculation = r(:,1:T);
end


%%%RL scenario: Using paramters of alpha and beta using reinforcement learning...
%to calculate the message rate for each interval
function rate_calculation = rate_RL(T,K,r_initial,rg)
r  = zeros(K,T);    %Initializing the message rates for K users

%Setting the initial message rate for existing users
K_var   = max(size(r_initial));

%Initial values of alpha and beta
alpha   = 0.1;
beta    = 1/K;

if gt(K,K_var)
    r(1:K_var,1)    = r_initial(1:K_var);
    r(K_var+1:K,1)  = 1*rand(K - K_var,1); %Added users are randmly initialized
else
    r(1:K,1) = r_initial(1:K);
end

rc(1:T) = zeros;    %The measured CBR
rc(1)   = 1100;     %Initlialize rc

for t = 2:T
   e   = rg - rc(t-1);
   r(:,t)      = (1 - alpha) * r(:,t-1) + beta * e;
   %Q-Learning algorithm implementation
   %Define the function to minimize the error function which will provide
   %optimal values for alpha and beta
   Q = @(v)(rg - ((1 - v(1) - K*v(2))*rc(t-1) + K*v(2)*rg));
   A       = [0,1];  %Less than equal to
   b       = 1.1/K;    %Beta value based on stability criteria
   Aeq     = [];       %Equivalency
   beq     = [];       %Equivalency
   lb      = [0.1,0];  %Lower bound
   ub      = [0.9,1.1/K];  %Upper bound
   v0      = [alpha,beta];    %Starting values

   opt = optimset('Display','off');
   v   = fmincon(Q,v0,A,b,Aeq,beq,lb,ub,[],opt); %Minimize the error function 

   %Re-set alpha and beta to calculated the optimal rc value
   alpha    = v(1);
   beta     = v(2);
   
   reward   = 100;	%Reward function
   l_rate   = 0.9;	%Learning rate
   dis      = 0.9;	%Discount factor
   rc_opt   = ((1 - alpha - K*beta)*rc(t-1) + K*beta*rg); %Optimal rc value
   
   %Temporal difference target
   TD_target  = l_rate*(reward + dis*rc_opt - rc(t-1));

   %New value of rc
   rc(t) = rc(t-1) + TD_target;
end

rate_calculation = r(:,1:T);
end