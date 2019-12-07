% script for loading and plotting the resutls of n_data study
clear all
clc

% files = dir('../results/net_size_study/*.mat');
files = dir('../results/net_size_study200/*.mat');

for i = 1:length(files)
    r(i) = load(strcat(files(i).folder,'/',files(i).name));
end

net_size = NaN(2,length(r));
net_size(:) = [r(:).net_hidden_size];

[~,I] = sort(net_size(1,:));

net_size = net_size(:,I);
r = r(I);

rms_error_trials = [r(:).final_rms_error];
rms_error_trials_uc = [r(:).final_rms_error_uc];

%% get unique n_data numbers

[~, IA, IC] = unique(net_size(1,:));
u_net_size = net_size(:,IA);
n_trials = diff(IA);
if ~all(n_trials == n_trials(1))        % check if same number of trials have been run for each experiment
    error('Different number of trials for each experiment')
end
n_exp = length(u_net_size);
rms_error = NaN(n_exp,n_trials(1));
rms_error_uc = NaN(n_exp,n_trials(1));

%%

for i = 1:length(u_net_size)
   I = i==IC;
   rms_error(i,:) = rms_error_trials(I);
    rms_error_uc(i,:) = rms_error_trials_uc(I);
end

%%
fontsize = 20;

% inds = 1:24;
% inds = [1:7, 9:2:19, 20 21 22 23];
inds = 1:length(u_net_size);
figure(1)
set(gcf,'Position',[34         487        1080         438])
subplot 121
boxplot(log(rms_error(inds,:).'),sum(u_net_size(:,inds),1))
set(gca,'FontSize',fontsize/1.75);
xlabel('Number of neurons','Interpreter','latex','FontSize',fontsize)
ylabel('log rms error','Interpreter','latex','FontSize',fontsize)
title('Constrained Neural Network','Interpreter','latex','FontSize',fontsize)
ylim([-4.1 0.5])
grid on

subplot 122
boxplot(log(rms_error_uc(inds,:).'),sum(u_net_size(:,inds),1))
set(gca,'FontSize',fontsize/1.75);
xlabel('Number of neurons','Interpreter','latex','FontSize',fontsize)
ylabel('log rms error','Interpreter','latex','FontSize',fontsize)
title('Standard Neural Network','Interpreter','latex','FontSize',fontsize)
ylim([-4.1 0.5])
grid on

figure(2)
plot(sum(u_net_size,1),log(mean(rms_error,2)))
hold on
plot(sum(u_net_size,1),log(mean(rms_error_uc,2)))
hold off
legend('Constrained NN','Standard NN')
xlabel('Number of measurements')
ylabel('log rms error')

% plot([r(:).n_data])



