% script for loading and plotting the resutls of n_data study
clear all
clc

files = dir('../results/mag_data_n_study/*.mat');

for i = 1:length(files)
    r(i) = load(strcat(files(i).folder,'/',files(i).name));
end

n_data = [r(:).n_data];

[n_data,I] = sort(n_data);
r = r(I);

rms_error_trials = [r(:).final_rms_error];
rms_error_trials_uc = [r(:).final_rms_error_uc];

%% get unique n_data numbers
[u_n_data, IA, IC] = unique(n_data);
n_trials = diff(IA);
if ~all(n_trials == n_trials(1))        % check if same number of trials have been run for each experiment
    error('Different number of trials for each experiment')
end
n_exp = length(u_n_data);
rms_error = NaN(n_exp,n_trials(1));
rms_error_uc = NaN(n_exp,n_trials(1));

%%

for i = 1:length(u_n_data)
   I = i==IC;
   rms_error(i,:) = rms_error_trials(I);
    rms_error_uc(i,:) = rms_error_trials_uc(I);
end

%%
fontsize = 20;

figure(1)
set(gcf,'Position',[34         487        1080         438])
subplot 121
boxplot(log(rms_error.'),u_n_data)
set(gca,'FontSize',fontsize/1.75);
xlabel('Number of measurements','Interpreter','latex','FontSize',fontsize)
ylabel('log rms error','Interpreter','latex','FontSize',fontsize)
title('Constrained Neural Network','Interpreter','latex','FontSize',fontsize)
ylim([-4.1 0.5])
grid on

subplot 122
boxplot(log(rms_error_uc.'),u_n_data)
set(gca,'FontSize',fontsize/1.75);
xlabel('Number of measurements','Interpreter','latex','FontSize',fontsize)
ylabel('log rms error','Interpreter','latex','FontSize',fontsize)
title('Standard Neural Network','Interpreter','latex','FontSize',fontsize)
ylim([-4.1 0.5])
grid on

figure(2)
plot(u_n_data,log(mean(rms_error,2)))
hold on
plot(u_n_data,log(mean(rms_error_uc,2)))
hold off
legend('Constrained NN','Standard NN')
xlabel('Number of measurements')
ylabel('log rms error')

% plot([r(:).n_data])



