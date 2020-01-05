% Figures in our paper are plotted with MATLAB
clear all

load('macer.mat')
hd = hard;
clear hard
load('cohen.mat')
hc = hard;
clear hard
load('salman.mat')
h1 = hard;
clear hard

hd(hd<0) = 0;
hc(hc<0) = 0;
h1(h1<0) = 0;

n = 100;
lo = 0;
hi = 2.0; % 1.0 for sigma=0.25, 2.0 for sigma=0.5 and 4.0 for sigma=1.0

x = linspace(lo, hi, n);
y = zeros(1, n);
for i = 1:n
    y(i) = sum(hd > x(i));
end
k = length(hd);
plot(x,y/k,'r-','linewidth',3)
hold on


for i = 1:n
    y(i) = sum(hc > x(i));
end
k = length(hc);
plot(x,y/k,'b--','linewidth',3)


for i = 1:n
    y(i) = sum(h1 > x(i));
end
k = length(h1);
plot(x,y/k,'m:','linewidth',3)


ylim([0 1])
lgd = legend('MACER-0.50','Cohen-0.50','Salman-0.50');
xl = xlabel('$l_2$ radius');
xl.Interpreter = 'latex';
ylabel('Certified accuracy');
set(gca,'fontsize',18)
set(gcf,'position',[100,100,500,400])