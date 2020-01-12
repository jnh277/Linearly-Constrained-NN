% script for loading and plotting the resutls of n_data study
clear all
clc

files = dir('../results/dims_study2/*.mat');

for i = 1:length(files)
    r(i) = load(strcat(files(i).folder,'/',files(i).name));
end


dims = [r(:).dims];

[~,I] = sort(dims);

dims = dims(I);
r = r(I);

val_loss_trials = [r.val_loss];
% rms = [r.rms];
val_loss_uc_trials = [r.val_loss_uc];
% rms_uc = [r.rms_)];

%% get unique n_data numbers

[~, IA, IC] = unique(dims);
u_dims = dims(:,IA);
n_trials = diff(IA);
if ~all(n_trials == n_trials(1))        % check if same number of trials have been run for each experiment
    error('Different number of trials for each experiment')
end
n_exp = length(u_net_size);
final_val_loss = NaN(n_exp,n_trials(1));
final_val_loss_uc = NaN(n_exp,n_trials(1));

%%

for i = 1:length(u_net_size)
   I = i==IC;
   final_val_loss(i,:) = val_loss_trials(end,I);
    final_val_loss_uc(i,:) = val_loss_uc_trials(end,I);
end

%%
fontsize = 26;

% inds = 1:24;
% inds = [1:7, 9:2:19, 20 21 22 23];
figure(1)
set(gcf,'Position',[34         487        1080         438])
subplot 121
boxplot(log(final_val_loss.'),sum(u_net_size,1))
set(gca,'FontSize',11.5);
xlabel('Number of neurons','Interpreter','latex','FontSize',fontsize)
ylabel('log rms error','Interpreter','latex','FontSize',fontsize)
title('Constrained Neural Network','Interpreter','latex','FontSize',fontsize)
ylim([-7.8 -6.55])
grid on

subplot 122
boxplot(log(final_val_loss_uc.'),sum(u_net_size,1))
set(gca,'FontSize',11.5);
xlabel('Number of neurons','Interpreter','latex','FontSize',fontsize)
ylabel('log rms error','Interpreter','latex','FontSize',fontsize)
title('Standard Neural Network','Interpreter','latex','FontSize',fontsize)
ylim([-7.8 -6.55])
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
mean(final_val_loss.',1)
std(final_val_loss.')

mean(final_val_loss_uc.',1)
std(final_val_loss_uc.')
