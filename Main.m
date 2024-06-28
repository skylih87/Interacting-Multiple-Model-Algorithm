clear; close all;
%% Simple demonstration for IMM using velocity and acceleration models
% Dimensionality of the state space
dims = 4;
% Space for models
ind = cell(1,3); % model index
F = cell(1,3);  % system matrix
Q = cell(1,3); % variance of system noise
H = cell(1,3); % measurement matrix
R = cell(1,3); % variance of measurement noise
% Stepsize
dt = 1;
%% generate system model
omega = 0.04; % turn rage [rad/s]

%model 1
ind{1} = [1 2 3 4]'; % Indexes of state components
% Transition matrix for the continous-time acceleration model.
F{1} = [1 dt 0 0;
        0 1 0 0;
        0 0 1 dt;
        0 0 0 1];
% Noise effect matrix for the continous-time system.
% system noise
B0= [ (dt^2)/2; dt ]; % position, velocity
L1= [ B0 zeros(2,1); zeros(2,1) B0 ];
% V= zeros(size(B,1),size(T_r,2));
sigma_v = 1;
V=L1*randn(size(L1,2),100); % nosie
Q{1}=diag([1 0.01 1 0.01]); % nosie covariance

% model 2
ind{2} = [1 2 3 4]';
% Transition matrix for the continous-time velocity model.
F{2} = [1 sin(omega)/omega 0 (cos(omega)-1)/omega;
      0 cos(omega) 0 -sin(omega);
      0 (1-cos(omega))/omega 1 sin(omega)/omega;
      0 sin(omega) 0 cos(omega)];
% Noise effect matrix for the continous-time system.
L2 = L1 ;
  % Process noise variance
Q{2} = Q{1};

% model 3
ind{3} = [1 2 3 4]';
% Transition matrix for the continous-time velocity model.
F{3} = [1 sin(-omega)/-omega 0 (cos(-omega)-1)/-omega;
      0 cos(-omega) 0 -sin(-omega);
      0 (cos(-omega)-1)/-omega 1 sin(-omega)/-omega;
      0 sin(-omega) 0 cos(-omega)];
% Noise effect matrix for the continous-time system.
L3 = L1 ;
  % Process noise variance
Q{3} = Q{1};

%% generate measurement model
h = [ 1 0 0 0 ;0 0 1 0]; hdims = 2;
H{1} = h; H{2} = h; H{3} = h;
W0 = diag([sqrt(400) sqrt(400)]);
r = W0*W0';
R{1} = r; R{2} = r; R{3} = r; 
%% generate the data
n =500;
Y = zeros(hdims,n); % measurement value
X_r = zeros(dims,n); % true state value
X_r(:,1) = [100 25 100 35]'; % initial position (position, velocity)
mstate = zeros(1,n);

% Forced mode transitions 
mstate(1:100) = 1;
mstate(101:200) = 2;
mstate(201:300) = 3;
mstate(301:400) = 2;
mstate(401:500) = 3;
mstate(501:600) = 1;

% !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! %
% random seed - do not change      %
rng(3);
% !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! %
for i = 2:n
    st = mstate(i);
    X_r(ind{st},i) = F{st}*X_r(ind{st},i-1) + gauss_rnd(zeros(size(F{st},1),1), Q{st});
end
% Generate the measurements.
for i = 1:n
    Y(:,i) = H{mstate(i)}*X_r(ind{mstate(i)},i) + gauss_rnd(zeros(size(Y,1),1), R{mstate(i)});
end

%% Filter %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
M1 = X_r(:,1); % initial state estimate
P1 = diag([0.1^2 0.1^2 0.1^2 0.1^2]); % initial covariance
% model1 Kalman filter %
for t = 1:n
    [M1, P1] = kf_predict(M1,P1,F{1},Q{1});
    [M1, P1] = kf_update(M1,P1,Y(:,t),H{1},R{1}); 
    MM1(:,t) = M1;
    PP1(:,:,t) = P1;
end
% model2 Kalman filter %
M2 = X_r(:,1); % initial state estimate
P2 = diag([0.1^2 0.1^2 0.1^2 0.1^2]); % initial covariance
for t = 1:n
    [M2, P2] = kf_predict(M2,P2,F{2},Q{2});
    [M2, P2] = kf_update(M2,P2,Y(:,t),H{2},R{2}); 
    MM2(:,t) = M2;
    PP2(:,:,t) = P2;
end
% model3 Kalman filter %
M3 = X_r(:,1); % initial state estimate
P3 = diag([0.1^2 0.1^2 0.1^2 0.1^2]); % initial covariance
for t = 1:n
    [M3, P3] = kf_predict(M3,P3,F{3},Q{3});
    [M3, P3] = kf_update(M3,P3,Y(:,t),H{3},R{3}); 
    MM3(:,t) = M3;
    PP3(:,:,t) = P3;
end

% IMM filter %
MM{1} = X_r(:,1); % initial state estimate
PP{1} = diag([0.1^2 0.1^2 0.1^2 0.1^2]); % initial covariance
MM{2} = X_r(:,1); % initial state estimate
PP{2} = diag([0.1^2 0.1^2 0.1^2 0.1^2]); % initial covariance
MM{3} = X_r(:,1); % initial state estimate
PP{3} = diag([0.1^2 0.1^2 0.1^2 0.1^2]); % initial covariance

mu_ip = [1/3 1/3 1/3];
cp_ij = [0.95 0.025 0.025; % Transition probability matrix (markov chain)
        0.025 0.95 0.025;
        0.025 0.025 0.95];

for t = 1:n
    [X_p,P_p,c_j] = imm_predict(MM,PP,mu_ip,cp_ij,ind,dims,F,Q);
    [MM,PP,mu_ip,m,P] = imm_update(X_p,P_p,c_j,ind,dims,Y(:,t),H,R);
    cmean(:,t)   = m;
    cmean_P(:,:,t) = P;
    cMU(:,t)   = mu_ip';
end    

% propose IMM filter %
MM{1} = X_r(:,1); % initial state estimate
PP{1} = diag([0.1^2 0.1^2 0.1^2 0.1^2]); % initial covariance
MM{2} = X_r(:,1); % initial state estimate
PP{2} = diag([0.1^2 0.1^2 0.1^2 0.1^2]); % initial covariance
MM{3} = X_r(:,1); % initial state estimate
PP{3} = diag([0.1^2 0.1^2 0.1^2 0.1^2]); % initial covariance

mu_ip = [1/3 1/3 1/3];
p_ij = [0.95 0.025 0.025; % Transition probability matrix (markov chain)
        0.025 0.95 0.025;
        0.025 0.025 0.95];
    
for t = 1:n
    [X_p,P_p,c_j,MU_ij] = imm_predict_stable(MM,PP,mu_ip,p_ij,ind,dims,F,Q);
    [MM,PP,mu_ip,m,P,LH] = imm_update_stable(X_p,P_p,c_j,ind,dims,Y(:,t),H,R);
    lambda(:,t) = LH;
    mean(:,t)   = m;
    mena_P(:,:,t) = P;
    MU(:,t)   = mu_ip';

    if t >= 2
        delta(:,t) = [1/(1+exp(-(MU(1,t) - MU(2,t-1) - MU(3,t-1))*1));
                      1/(1+exp(-(MU(2,t) - MU(3,t-1) - MU(1,t-1))*1));
                      1/(1+exp(-(MU(3,t) - MU(1,t-1) - MU(2,t-1))*1))];
        delta_MU = MU(:,t) - MU(:,t-1);
        f_i(:,t) = 1./(1-(delta_MU));
        p_ij = p_ij.*f_i(:,t)' - p_ij.*delta(:,t);
        for i = 1:3
            p_ij(i,:) = p_ij(i,:)./sum(p_ij(i,:));
        end
    end
    checkp_11(t)=p_ij(1,1);
    checkp_12(t)=p_ij(1,2);
    checkp_13(t)=p_ij(1,3);
    checkp_21(t)=p_ij(2,1);
    checkp_22(t)=p_ij(2,2);
    checkp_23(t)=p_ij(2,3);
    checkp_31(t)=p_ij(3,1);
    checkp_32(t)=p_ij(3,2);
    checkp_33(t)=p_ij(3,3);
end
% compare %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ce_i = X_r-cmean;
cRMS_i = (ce_i(1,:).^2+ce_i(3,:).^2);

e_p = X_r-mean;
RMS_p = (e_p(1,:).^2+e_p(3,:).^2);

cRMS_iv = (ce_i(2,:).^2+ce_i(4,:).^2);
RMS_pv = (e_p(2,:).^2+e_p(4,:).^2);

Position_Error_CIMM = sqrt(sum(cRMS_i)/n)
Position_Error_PIMM = sqrt(sum(RMS_p)/n)
Velocity_Error_CIMM = sqrt(sum(cRMS_iv)/n)
Velocity_Error_PIMM = sqrt(sum(RMS_pv)/n)

%% plotting %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(1); hold on; grid on; axis equal;
plot(X_r(1,:),X_r(3,:),'k','linewidth',2)

plot(cmean(1,:),cmean(3,:),'r:','linewidth',2)
plot(mean(1,:),mean(3,:),'b:','linewidth',2)
xlabel('x [m]');ylabel('y [m]');
legend('True Tracject','IMM','Proposed','FontSize',8)

figure(2); 
set(figure(2),'Position', [680 158 1000 800]); 
subplot(3,1,1); hold on; grid on;
plot(cMU(1,:),'r','linewidth',2);
plot(MU(1,:),'b','linewidth',2);
ylabel('CV Model Probability');
legend('IMM','Proposed','FontSize',12)

subplot(3,1,2); hold on; grid on;
plot(cMU(2,:),'r','linewidth',2);
plot(MU(2,:),'b','linewidth',2);
ylabel('CLT Model Probability');

subplot(3,1,3); hold on; grid on;
plot(cMU(3,:),'r','linewidth',2);
plot(MU(3,:),'b','linewidth',2);
xlabel('Time');ylabel('CRL Model Probability');

figure(3);
set(figure(3),'Position', [680 158 900 500]); 
subplot(2,1,1); hold on; grid on;
plot(sqrt(cRMS_i),'r','linewidth',2); 
plot(sqrt(RMS_p),'b','linewidth',2);
ylabel('Position RMSE');
legend('IMM','Proposed','FontSize',12)

subplot(2,1,2);  hold on; grid on;
xlabel('Time');ylabel('Velocity RMSE');
plot(sqrt(cRMS_iv),'r','linewidth',2); 
plot(sqrt(RMS_pv),'b','linewidth',2);


figure(4); subplot(2,1,1); hold on; grid on;
plot(f_i(1,:),'b','linewidth',2); plot(f_i(2,:),'r','linewidth',2); plot(f_i(3,:),'g','linewidth',2);
ylabel('Polarizing TCF');
legend('CV Model','CLT Model','CRL Model','FontSize',10)
subplot(2,1,2); hold on; grid on;
plot(delta(1,:),'b','linewidth',2); plot(delta(2,:),'r','linewidth',2); plot(delta(3,:),'g','linewidth',2);
xlabel('Time');ylabel('Activating TCF');


figure(5); hold on; grid on;
plot(checkp_11,'linewidth',2);plot(checkp_12,'linewidth',2);plot(checkp_13,'linewidth',2);

plot(checkp_21,'linewidth',2);plot(checkp_22,'linewidth',2);plot(checkp_23,'linewidth',2);

plot(checkp_31,'linewidth',2);plot(checkp_32,'linewidth',2);plot(checkp_33,'linewidth',2);
legend('p_{11}','p_{12}','p_{13}','p_{21}','p_{22}','p_{23}','p_{31}','p_{32}','p_{33}','FontSize',12);xlabel('Time');ylabel('Probability');%title('CRT model TPM')