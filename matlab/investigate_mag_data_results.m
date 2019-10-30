clear all
clc

files = dir('../results/mag_data_tests2/*.mat');

for i = 1:length(files)
    r(i) = load(strcat(files(i).folder,'/',files(i).name));
    constrained_val_loss(i) = r(i).val_loss(end);
    standard_val_loss(i) = r(i).val_loss_uc(end);
end


% constrained_val_loss = deal(r(:).val_loss(end));
% standard_val_loss = [r(:).val_loss(end)];


figure(1)
clf
plot(log(constrained_val_loss))
hold on
plot(log(standard_val_loss))
hold off


n_train = [r(:).n_train];

ind = find(n_train == 5000);

figure(2)
clf
plot(ind,log(constrained_val_loss(ind)))
hold on
plot(ind,log(standard_val_loss(ind)))
hold off