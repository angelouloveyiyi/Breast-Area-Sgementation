I = imread('trans\0840.png');
%I = rgb2gray(I);
N = 30;
Data = zeros(N+1,2);
% out = zeros(1,N+1);
% cliplimit = zeros(1,N+1);
M = 100;
for i = 0:N
    Data(i+1,2) = i/M;
    Img = adapthisteq(I,'cliplimit',Data(i+1,2),'Distribution','rayleigh');
    Data(i+1,1) = entropy(Img);
end
figure(1)
plot(Data(:,2),Data(:,1),'r.');
xlabel('clip limit')
ylabel('entropy')
title('Entropy vs Clip Limit')
hold on;
% fit y = c(1)*exp(-lam(1)*x) + c(2)*exp(-lam(2)*t);
t = Data(:,2);
y = Data(:,1);
% x(1) = c(1) = 1;
% x(2) = lam(1) = 1;
% x(3) = c(2) = 1;
% x(4) = lam(2) = 0;
 F = @(x,xdata)x(1)*exp(-x(2)*xdata) + x(3)*exp(-x(4)*xdata);
 x0 = [0.01 0.01 0.01 0];
[x,resnorm,~,exitflag,output] = lsqcurvefit(F,x0,t,y);

plot(t,F(x,t))

hold off;

syms T
X(T) = T*1/M;
f(T) = x(1)*exp(x(2)*T*1/M)+x(3)*exp(x(4)*T*1/M);

K(T) = abs((diff(X(T))*diff(f(T),2))/((1/M)^2*sqrt(1^2+diff(f(T)^2)^3)));
for T = 0:N
   
   C(1,T+1) = K(T);
   
end
figure(2)
plot(C)
[h,T] = find(C==max(C));
ylabel('maximum curvature')
xlabel('time')
title('Maximum Curvature')
CL = T*1/M;
ImgRes = adapthisteq(I,'cliplimit',CL,'Distribution','rayleigh');
figure(3)
imshow(ImgRes)
title('CLAHE Image')








