% script for loading and plotting the resutls of n_data study
clear all
clc

files = dir('../results/pointObsStudy2/*.mat');

for i = 1:length(files)
    r(i) = load(strcat(files(i).folder,'/',files(i).name));
end

w= [r(:).constraint_weighting];

[w,I] = sort(w);
r = r(I);

rms_error_trials = [r(:).final_rms_error];
c_mae_trials = [r(:).c_mae];
c_mse_trials = [r(:).c_mse];

%% get unique n_data numbers
[u_w, IA, IC] = unique(w);
n_trials = diff(IA);
if ~all(n_trials == n_trials(1))        % check if same number of trials have been run for each experiment
    error('Different number of trials for each experiment')
end
n_exp = length(u_w);
rms_error = NaN(n_exp,n_trials(1));
c_mae = NaN(n_exp,n_trials(1));
c_mse = NaN(n_exp,n_trials(1));

%%

for i = 1:length(u_w)
    I = i==IC;
    rms_error(i,:) = rms_error_trials(I);
    c_mae(i,:) = c_mae_trials(I);
    c_mse(i,:) = c_mse_trials(I);
end


%% comparison values
constrained_rmse = exp(-3.491);     % modal value

%%
fontsize = 26;

figure(1)
clf
subplot 211
plot(u_w,mean(rms_error.^2,2))

subplot 212
plot(u_w,mean(c_mae,2))



figure(3)
clf
subplot 121
boxplot(log(rms_error).',u_w);
ylim([-3.75 -1])
h = gca;
xtk = h.XTick;
hold on
plot(xtk, xtk*0+log(constrained_rmse),'--','LineWidth',2);
hold off
xlabel('Constraint weighting: $\lambda$','Interpreter','Latex','FontSize',fontsize)
ylabel('log RMSE','FontSize',fontsize,'Interpreter','Latex')


subplot 122
boxplot(c_mae.',u_w)
ylim([-0.05 0.47])
h = gca;
xtk = h.XTick;
hold on
plot(xtk, xtk*0,'--','LineWidth',2);
hold off
xlabel('Constraint weighting: $\lambda$','Interpreter','Latex','FontSize',fontsize)
ylabel('mean absolute constraint violation','FontSize',fontsize,'Interpreter','Latex')



% hold on
% plot(u_w,c_mae)
% hold off
% fontsize = 26;
% 
% figure(1)
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



