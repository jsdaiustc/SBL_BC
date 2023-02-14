% The dataset is obtained from Laurent Duval (2020). BEADS Baseline Estimation And Denoising with Sparsity (https://www.mathworks.com/matlabcentral/fileexchange/49974-beads-baseline-estimation-and-denoising-with-sparsity), MATLAB Central File Exchange. Retrieved February 29, 2020. 

clear
close all

load noise.mat
load chromatogram.mat;
y=X(:,5)+ noise;

tic;
[w,A,b]=SBL_BC(y);
toc
s=A*w;

figure(1);plot(y,'b');hold on; plot(s ,'r'); plot(b,'g'); hold off;
legend('original spectrum','estimated spectrum','estimated baseline')
title('SBL-BC')
