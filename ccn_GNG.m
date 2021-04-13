

clc;
clear all;

W=127;
a=1000;
P=10^(-3);


if (W>=((2*a)+1))
    U=(1-P)/(1+(2*a*P));
elseif (W<((2*a)+1))
    U=(W*(1-P))/(((2*a)+1)*(1-P+(W*P)));
end
    
disp(U)        
