% script for loading and plotting the resutls of n_data study
clear all
clc

files = dir('../results/mag_data_n_study/*.mat');
% files = dir('../results/mag_data_n_study/exp_8000*.mat');
% for i = 1:length(files)
%     r = load(strcat(files(i).folder,'/',files(i).name));
%     nums = sscanf(files(i).name, 'exp_%d_trial_%d.mat');
%     trials(i) = nums(2);
% end


for i = 1:length(files)
    r(i) = load(strcat(files(i).folder,'/',files(i).name));
end

n_data = [r(:).n_train];

[n_data,I] = sort(n_data);
r = r(I);

val_loss_trials = [r.val_loss];
val_loss_uc_trials = [r.val_loss_uc];

%% get unique n_data numbers
[u_n_data, IA, IC] = unique(n_data);
n_trials = diff(IA);
if ~all(n_trials == n_trials(1))        % check if same number of trials have been run for each experiment
    error('Different number of trials for each experiment')
end
n_exp = length(u_n_data);
final_val_loss = NaN(n_exp,n_trials(1));
final_val_loss_uc = NaN(n_exp,n_trials(1));

%%

for i = 1:length(u_n_data)
   I = i==IC;
   final_val_loss(i,:) = val_loss_trials(end,I);
    final_val_loss_uc(i,:) = val_loss_uc_trials(end,I);
end

%%
fontsize = 20;

figure(1)
set(gcf,'Position',[34         487        1080         438])
subplot 121
boxplot(log(final_val_loss.'),u_n_data)
set(gca,'FontSize',fontsize/1.75);
xlabel('Number of measurements','Interpreter','latex','FontSize',fontsize)
ylabel('log rms validation loss','Interpreter','latex','FontSize',fontsize)
title('Constrained Neural Network','Interpreter','latex','FontSize',fontsize)
ylim([-8 -5])
grid on

subplot 122
boxplot(log(final_val_loss_uc.'),u_n_data)
set(gca,'FontSize',fontsize/1.75);
xlabel('Number of measurements','Interpreter','latex','FontSize',fontsize)
ylabel('log rms validation loss','Interpreter','latex','FontSize',fontsize)
title('Standard Neural Network','Interpreter','latex','FontSize',fontsize)
ylim([-8 -5])
grid on

figure(2)
plot(u_n_data,log(mean(final_val_loss,2)))
hold on
plot(u_n_data,log(mean(final_val_loss_uc,2)))
hold off
legend('Constrained NN','Standard NN')
set(gca,'FontSize',fontsize/1.75);
xlabel('Number of measurements','Interpreter','latex','FontSize',fontsize)
ylabel('log rms validation loss','Interpreter','latex','FontSize',fontsize)

mean(final_val_loss.',1)
std(final_val_loss.')

mean(final_val_loss_uc.',1)
std(final_val_loss_uc.')

% plot([r(:).n_data])
%%

% data = [final_val_loss.', final_val_loss_uc.'];
% cat1 = repmat(u_n_data,1,2);
% mod = [repmat({'M1'},1,length(u_n_data)),repmat({'M2'},1,length(u_n_data))];
% 
% figure(3)
% boxplot(log(data),{cat1,mod},'factorgap',[5 0],'labelverbosity','minor','colors',repmat('rb',1,12));


