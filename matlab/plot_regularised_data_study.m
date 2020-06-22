% script for loading and plotting the resutls of n_data study
clear all
clc

% files = dir('../results/n_data_study_reg01/*.mat');      
% files = dir('../results/n_data_study_reg001/*.mat');      
files = dir('../results/n_data_study_reg0001/*.mat');       % -2.929
% files = dir('../results/n_data_study_reg00001/*.mat');    
% files = dir('../results/n_data_study_reg0002/*.mat');     % -2.865
% files = dir('../results/n_data_study_reg00008/*.mat');     % -2.899

for i = 1:length(files)
    r(i) = load(strcat(files(i).folder,'/',files(i).name));
end

n_data = [r(:).n_data];

[n_data,I] = sort(n_data);
r = r(I);

rms_error_trials = [r(:).final_rms_error];
rms_error_trials_reg = [r(:).final_rms_error_reg];
rms_error_trials_uc = [r(:).final_rms_error_uc];
rms_error_trials_uc_reg = [r(:).final_rms_error_uc_reg];

%% get unique n_data numbers
[u_n_data, IA, IC] = unique(n_data);
n_trials = diff(IA);
if ~all(n_trials == n_trials(1))        % check if same number of trials have been run for each experiment
    error('Different number of trials for each experiment')
end
n_exp = length(u_n_data);
rms_error = NaN(n_exp,n_trials(1));
rms_error_uc = NaN(n_exp,n_trials(1));
rms_error_reg = NaN(n_exp,n_trials(1));
rms_error_uc_reg = NaN(n_exp,n_trials(1));

%%

for i = 1:length(u_n_data)
% for i = 1:10
   I = i==IC;
   rms_error(i,:) = rms_error_trials(I);
   rms_error_reg(i,:) = rms_error_trials_reg(I);
    rms_error_uc(i,:) = rms_error_trials_uc(I);
    rms_error_uc_reg(i,:) = rms_error_trials_uc_reg(I);
end

%%
fontsize = 26;

figure(3)
clf
plot(u_n_data,mean(log(rms_error),2),'--','LineWidth',2,'Color',[0, 0.4470, 0.7410])
hold on
plot(u_n_data,mean(log(rms_error_reg),2),'LineWidth',2,'Color',[0, 0.4470, 0.7410])
plot(u_n_data,mean(log(rms_error_uc),2),'--','LineWidth',2,'Color',[0.8500, 0.3250, 0.0980])
plot(u_n_data,mean(log(rms_error_uc_reg),2),'LineWidth',2,'Color',[0.8500, 0.3250, 0.0980])
hold off
set(gca,'FontSize',11.5);
hl = legend('Constrained neural network','Regularised constrained neural network','Standard neural network','Regularised standard neural network');
set(hl,'FontSize',16,'Interpreter','Latex')
% title('with weight decay = 0.005','FontSize',fontsize,'Interpreter','Latex')
xlabel('Number of measurements','FontSize',fontsize,'Interpreter','Latex')
ylabel('log RMSE','FontSize',fontsize,'Interpreter','Latex')
%%

figure(4)
boxplot(log(rms_error.'),u_n_data)

% figure(2)
% clf
% set(gcf,'Position',[34         487        1080         438])
% subplot 121
% boxplot(log(rms_error.'),u_n_data)
% set(gca,'FontSize',11.5);
% xlabel('Number of measurements','Interpreter','latex','FontSize',fontsize)
% ylabel('log RMSE','Interpreter','latex','FontSize',fontsize)
% title('Constrained Neural Network','Interpreter','latex','FontSize',fontsize)
% ylim([-4.1 0.5])
% grid on
% 
% subplot 122
% boxplot(log(rms_error_uc.'),u_n_data)
% set(gca,'FontSize',11.5);
% xlabel('Number of measurements','Interpreter','latex','FontSize',fontsize)
% yh = ylabel('log RMSE','Interpreter','latex','FontSize',fontsize);
% % set(yh,'Position',get(yh,'Position')+[0.05 0.0 0.0])
% title('Standard Neural Network','Interpreter','latex','FontSize',fontsize)
% ylim([-4.1 0.5])
% grid on
% p = get(gca,'Position');
% set(gca,'Position',p - [0.05 0 0 0])


% figure(2)
% plot(u_n_data,log(mean(rms_error,2)))
% hold on
% plot(u_n_data,log(mean(rms_error_uc,2)))
% hold off
% legend('Constrained NN','Standard NN')
% xlabel('Number of measurements')
% ylabel('log rms error')

% plot([r(:).n_data])



