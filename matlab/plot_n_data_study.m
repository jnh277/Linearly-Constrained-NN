% script for loading and plotting the resutls of n_data study

files = dir('../results/n_data_study/*.mat');

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
figure(7)
set(gcf,'Position',[34         487        1080         438])
subplot 121
boxplot(log(rms_error.'),u_n_data)
xlabel('Number of measurements')
ylabel('log rms error')
title('Constrained Neural Network')
ylim([-4.1 0.5])

subplot 122
boxplot(log(rms_error_uc.'),u_n_data)
xlabel('Number of measurements')
ylabel('log rms error')
title('Standard Neural Network')
ylim([-4.1 0.5])

figure(8)
plot(u_n_data,log(mean(rms_error,2)))
hold on
plot(u_n_data,log(mean(rms_error_uc,2)))
hold off
legend('Constrained NN','Standard NN')
xlabel('Number of measurements')
ylabel('log rms error')

% plot([r(:).n_data])



