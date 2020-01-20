% script for loading and plotting the resutls of n_data study
clear all
clc

% files = dir('../results/dims_study2/*.mat'); old
% files = dir('../results/dims_study/*.mat');
% files = [files;dir('../results/dims_study/exp_4*.mat');]
files = dir('../results/dims_study3/*.mat');

for i = 1:length(files)
    r(i) = load(strcat(files(i).folder,'/',files(i).name));
end


dims = [r.dims];

[~,I] = sort(dims);

dims = dims(I);
r = r(I);



% val_loss_trials = [r.val_loss]; this is wrong
% rms = [r.rms];
% val_loss_uc_trials = [r.val_loss_uc];
% rms_uc = [r.rms_uc];


% ind = logical((dims == 3) + (dims == 4) + (dims == 7));
% dims(ind) = [];
% val_loss_trials(ind) = [];
% val_loss_uc_trials(ind) = [];
%% get unique n_data numbers

[~, IA, IC] = unique(dims);
u_dims = dims(:,IA);
n_trials = diff(IA);

% ind = n_trials ~= 20


if ~all(n_trials == n_trials(1))        % check if same number of trials have been run for each experiment
    error('Different number of trials for each experiment')
end
n_exp = length(u_dims);
% final_val_loss = NaN(n_exp,n_trials(1));
% final_val_loss_uc = NaN(n_exp,n_trials(1));


rms = NaN(n_trials(1),n_exp);
rms_uc =NaN(n_trials(1),n_exp);

rms(:) = [r.rms].';
rms_uc(:) = [r.rms_uc].';

%%

for i = 1:length(u_dims)
   I = i==IC;
%    final_val_loss(i,:) = val_loss_trials(end,I);
%     final_val_loss_uc(i,:) = val_loss_uc_trials(end,I);
%     final_val_loss(i,:) = val_loss_trials(I);
end

%%
fontsize = 26;

% inds = 1:24;
% inds = [1:7, 9:2:19, 20 21 22 23];
figure(1)
set(gcf,'Position',[34         487        1080         438])
subplot 121
boxplot(log(rms),u_dims)
set(gca,'FontSize',11.5);
xlabel('Number of output dimensions','Interpreter','latex','FontSize',fontsize)
ylabel('log rms error','Interpreter','latex','FontSize',fontsize)
title('Constrained Neural Network','Interpreter','latex','FontSize',fontsize)
ylim([-4.5 -2.75])
grid on

subplot 122
boxplot(log(rms_uc),u_dims)
set(gca,'FontSize',11.5);
xlabel('Number of output dimensions','Interpreter','latex','FontSize',fontsize)
ylabel('log rms error','Interpreter','latex','FontSize',fontsize)
title('Standard Neural Network','Interpreter','latex','FontSize',fontsize)
ylim([-4.5 -2.75])
grid on
p = get(gca,'Position');
set(gca,'Position',p - [0.05 0 0 0])
%%
% figure(2)
% plot(sum(u_net_size,1),log(mean(final_val_loss,2)))
% hold on
% plot(sum(u_net_size,1),log(mean(final_val_loss_uc,2)))
% hold off
% legend('Constrained NN','Standard NN')
% xlabel('Number of measurements')
% ylabel('log rms error')

% plot([r(:).n_data])

%%
mean(rms.',1)
std(rms.')

mean(rms_uc.',1)
std(rms_uc.')
