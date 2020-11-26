%%=============================================================
%
%
% A LIM tutorial (partly derived from Oxford code (ref. L. Zanna))
% - Aneesh C. S. (Sept 2015)
%
%%=============================================================

clear all
close all

colors1=[ 'r';'g';'b';'c';'m'...
    ;'r';'g';'b';'c';'m'];

% size of the sample data matrix:
ND=6;

% Number of leading modes to keep (NM)
NM=6;

%length of sample timeseries
LEN=100;

% lead time for the lag covariance matrix (Ideally the LIM model should be
% insensitive to the length of tau, Newman 2007 J. Clim.)
tau=1;

% amplitude of random number
amp_rand=0.04;

%Sample coefficient matrix to test the LIM on
A = [
    0.6473    0.7794   -0.2384   -0.2514    0.2725    0.1545;
    0.3763   -0.0788    0.2912    0.0157    0.5366    0.4495;
    0.1415   -0.1115    0.6313    1.0658   -0.1281   -0.6122;
    0.2403   -0.6710   -0.5115   -1.2562    0.3314   -0.2627;
   -0.6190    0.8029    0.2052   -1.2039    0.8788    1.3340;
    0.2461   -0.5397   -0.1025   -0.5906    0.1451    0.6633  ];


xp = nan(ND,LEN);

% Create random data to initialize time series
xp(:,1)= randn(ND,1);  % random start point

for tt=1:LEN-1
  xp(:,tt+1)=A*xp(:,tt) + amp_rand*randn(ND,1);
end
xp_orig=xp;


%%%%%%%%%%%%%%%%%%%%%%%
% Compute temporal anomalies of the timeseries
%%%%%%%%%%%%%%%%%%%%%%%

for ii=1:ND
  xp(ii,:)=xp(ii,:)-mean(xp(ii,:),2);
end

% SVD decomposition of the anomalies time series
[U,S,V]=svd(xp',0);

% Keep on NM number of modes
x=U(:,1:NM)';
y=V(1:NM,1:NM)';

% Trace of eigenvalue matrix
tr=sum(diag(S).^2);

% fraction of variance explained
FVE = diag(S).^2 / tr  

% normalise time series and scale in singular values to retain variance
for ii=1:NM
  n=var(x(ii,:));
  x(ii,:) = x(ii,:)/n;
  y(ii,:) = y(ii,:)*n*S(ii,ii);
end

% check reconstructed timeseries ok
figure
subplot(3,1,1)
imagesc(xp');
colorbar;
ylabel('Raw data');
caxis([-5 5]);
set(gca,'Box','On','fontsize',14,'fontweight','b');

subplot(3,1,2);
imagesc(U*S*V');
colorbar;
ylabel('Full EOFs');
caxis([-5 5]);
set(gca,'Box','On','fontsize',14,'fontweight','b');

subplot(3,1,3);
imagesc(x'*y);
xlim([0.5 ND+0.5]);
colorbar;
caxis([-5 5]);
ylabel('Leading modes only')
set(gca,'Box','On','fontsize',14,'fontweight','b');
set(gcf,'color','w');

% weight x by sqrt(FVE) for Optimal Perturbation analysis
for ii=1:NM
  x(ii,:) = x(ii,:)*sqrt(FVE(ii));
end

%% ===============
% LIM Matrix
%%================

s=size(x);
Clag = zeros(s(1),s(1));

% Compute lag covariance matrix with lag tau
for ii=1:s(2)-tau
  Clag = Clag + (x(:,ii+tau)*(x(:,ii)'));
end

% Compute the zero lag covariance matrix
C = x*x' / (s(2)-1);

% Normalize lag covariance matrix
Clag = Clag / (s(2)-tau-1);

% Constructing the LIM matrix
B = logm(Clag/C)/tau; 

% Integrate the model for time tt

lprop = nan(NM,NM,LEN);

for tt=1:LEN
 lprop(:,:,tt) = expm(B*tt);
end

%%
% Eigenvalues and eigenvectors of Propagator matrix
kk=1;
for tt=0:1:LEN
  PROP = expm(B*tt);
  [q,w]=eigs(PROP'*PROP,NM);
  [aa,indds] = max(diag(w));
  allss =  diag(w);
  amp_energy(kk) = allss(indds);

  disp(sqrt(q(:,indds)'*q(:,indds)));
  q(:,indds) = q(:,indds) / sqrt(q(:,indds)'*q(:,indds));
  ev(:,kk) = q(:,indds);
  tim(kk)=tt;
  kk=kk+1;
end
%%
max_amp = max(amp_energy);
max_tim = tim(find(amp_energy==max_amp));


clear D1 V1 t
% %% original growth 
indx_t=0; dt=1;
sig_orig = nan(1,LEN);
for t=dt:dt:100
  indx_t=indx_t+1;
   [V1,D1]=eig((A^t)'*(A^t));
   sig_orig(indx_t) = max(diag(D1)); 
end; 

% --------
dt=1;
ta = [0:dt:100]; 
figure 
plot(tim,amp_energy,'r.','MarkerSize',15);
xlim([0 40])
ylim([0 max(sig_orig)+10])

hold on 
plot(ta,[1 sig_orig],'k+','MarkerSize',15);
legend('LIM','Orig','Location','NorthWest')
xlabel('Time','fontsize',20,'fontweight','b'); 
ylabel('|P(t)|','fontsize',20,'fontweight','b'); 
set(gca,'fontsize',20,'fontweight','b')

