function [mu_w,A,mu_b]=SBL_BC(Y)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Codes for H. Li, J. Dai, T. Pan, C. Chang, and H. C. So, "Sparse Bayesian learning approach for baseline correction"
% Written by Jisheng Dai

% Y:    input spectrum
% mu_w: estimated spectrum
% A:    returned measurement with off-grid updating 
% mu_b: estimated baseline
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% scale the data
[N,M]=size(Y);
Scale_y=max(Y(:))*10;
Y=Y/Scale_y;

%% select rho
if N<1000
    rho=10^5;
elseif N<2000
    rho=10^7;
else
    rho=10^8;
end

%% construct the measurement matrix A
rev=5;
upsilon_all=1./(ones(N,1)*rev);
wave_sample=[1:N]';
a_one= exp(  -  (( wave_sample- 1 ).^2)   /2  *upsilon_all(1)     );
a_one=a_one(find(a_one>10^(-2)));
length_sparse=length(a_one)-1;
a_one=[ a_one(end:-1:2); a_one ]';
A = spdiags( a_one(ones(N,1), :) , -length_sparse:length_sparse, N, N);  
A2=A'*A;

%% initializations
converged = false;
iter=0;
c_constant=1e-10;
d_constant=1e-10;
alpha0=1;
gamma=ones(N,1)*10^(6);  
D = diff(speye(N),2);
D2= D'*D;
W= speye(N) +  10^5* D2;
mu_b=W\Y;
maxiter=100;
position=[1:N];
etc=20;
mu_w=1000;

%% 
while ~converged
   
     %update mu_s
     mu_old=mu_w;
     Sigma_s=  inv(alpha0*A2 + spdiags(gamma,0,length(gamma),length(gamma)) );
     mu_w = Sigma_s * (alpha0 * (A' * (Y -mu_b ) )  );
     mu_w= (mu_w+ abs(mu_w))/2;
     
     %update alpha0
     resid=Y-A*mu_w-mu_b;
     z1= length(gamma)-  trace( Sigma_s*  spdiags(gamma,0,length(gamma),length(gamma)) ) ;
     term2=z1/alpha0;
     alpha0=( N*M + c_constant )/( d_constant +  ( norm(resid(:), 'fro')^2+   M*real(term2) )  );
     alpha0= min(alpha0,10^5);
      
     %update gamma
     t1= sum(abs(mu_w).^2,2)  +  M * real(diag(Sigma_s));
     tc=real(t1);
     c_k = c_constant +  M;
     d_k = d_constant +  tc;
     gamma=c_k./d_k;
     
     %update mu_b
     mu_b=(   (   speye(N) +   D2*rho )\(Y- A*mu_w));
      
     %update grid
     if iter>20
         sum_mu=sum(mu_w.*conj(mu_w),2);
         resid=Y-A*mu_w-mu_b;
         Pm=sum_mu;
         [~,sort_ind]=sort(Pm, 'descend');
         index_amp = sort_ind(1:etc);
         tempPS=A*  Sigma_s(:,index_amp) ;
         df=zeros(length(index_amp),1);
         for j=1:length(index_amp)
             ii=index_amp(j);
             ai=  exp(  -  (( wave_sample- position(ii) ).^2)  *  (upsilon_all(ii))/2    );
             mut=mu_w(ii,:);
             Sigmat=Sigma_s(:,ii);
             c1=mut*mut' +  M* Sigmat(ii);
             c1=abs(c1)*(-alpha0);
             Yti=resid +  A(:, ii)*mu_w(ii,:);
             c2=  M*(  tempPS(:,j) - A(:,ii)*Sigmat(ii) )  -Yti*(mut');
             c2= c2*(-alpha0);
             phii=A(:,ii);
             c3= -(( wave_sample- position(ii) ).^2) /2  ;
             tt1= (c3.*ai);
             f1= tt1'*phii*c1  +   tt1'*c2;
             f1= 2*real(f1);
             df(j)=f1;
         end
         sign_df=sign(df);
         ddff=sign_df/(rev*100);
         upsilon_all(index_amp) = upsilon_all(index_amp) +  ddff;
         A_active=zeros(size(A,1), etc);
         for ee=1:etc
             A_active(:,ee)= exp(  -  (( wave_sample- position(index_amp(ee)) ).^2)  *  (upsilon_all(index_amp(ee)).')/2    );
         end
         A_active(find(A_active<10^(-5)))=0;
         A(:,index_amp)=A_active;
         A2=A'*A;
     end
     
     % redcue the active size
     if iter==20-1 
        ind_remove=find(gamma>max(gamma)/2);
        gamma(ind_remove)=[];
        A(:,ind_remove)=[];
        upsilon_all(ind_remove)=[];
        position(ind_remove)=[];
        mu_w=1000;
        A2=A'*A;
     end

     % stopping criteria
     if norm(mu_w-mu_old)/norm(mu_w)<1e-3 && iter>10
         break
     end
     if iter >= maxiter
         converged = true;
     end
     iter = iter + 1;
   
end

%% output
mu_w=mu_w*Scale_y;
mu_b=mu_b*Scale_y;


