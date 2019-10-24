% rename files
clear all
clc

files = dir('../results/net_size_study/*.mat');

for i = 1:length(files)
    r = load(strcat(files(i).folder,'/',files(i).name));
    nums = sscanf(files(i).name, 'exp_%d_trial_%d.mat');
    save(['../results/net_size_study2/exp_' num2str(r.net_hidden_size(1)) '_trial_' num2str(nums(2))],'-struct','r')
    disp(num2str(i/length(files)))
end