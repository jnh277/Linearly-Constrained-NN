clear all
clc


load('../results/mag_plot_data.mat')

S = 4;
figure(1)
clf
q = quiver3(pos(:,1),pos(:,2),pos(:,3),mag_true(:,1),mag_true(:,2),mag_true(:,3),S);
hold on
q2 = quiver3(pos(:,1),pos(:,2),pos(:,3),m1pred,m2pred,m3pred,S);
hold off
xlabel('x')
ylabel('y')
zlabel('z')
% xlim([-1 0])
% ylim([-0.4 -0.2])
axis square

figure(2)
clf
q = quiver3(pos(:,1),pos(:,2),pos(:,3),mag_true(:,1)-m1pred,mag_true(:,2)-m2pred,mag_true(:,3)-m3pred,S);
hold on
scatter3(pos(1:n_train,1),pos(1:n_train,2),pos(1:n_train,3))
hold off
rms_err = sqrt(mean([mag_true(:,1)-m1pred;mag_true(:,2)-m2pred;mag_true(:,3)-m3pred].^2));

figure(3)
clf
q = quiver3(pos(:,1),pos(:,2),pos(:,3),mag_true(:,1)-mpreduc(:,1),mag_true(:,2)-mpreduc(:,2),mag_true(:,3)-mpreduc(:,3),S);

rms_err_uc = sqrt(mean([mag_true(:,1)-mpreduc(:,1);mag_true(:,2)-mpreduc(:,2);mag_true(:,3)-mpreduc(:,3)].^2));

%%
orig_data = load('../real_data/magnetic_field_data');
S = 0.1;
sc = 0.1;
pinds = 1:3:length(pos);
figure(4)
clf
hold on
q3 = quiver3(orig_data.pos(pinds,1),orig_data.pos(pinds,2),orig_data.pos(pinds,3),m1pred_o(pinds)*sc,m2pred_o(pinds)*sc,m3pred_o(pinds)*sc);
q3.AutoScale = 'off';
q3.LineWidth = 1.0;
plot3(orig_data.pos(:,1),orig_data.pos(:,2),orig_data.pos(:,3),'k')
% q = quiver3(pos(pinds,1),pos(pinds,2),pos(pinds,3),mag_true(pinds,1)*sc,mag_true(pinds,2)*sc,mag_true(pinds,3)*sc,'k');
% q.AutoScale = 'off';
q2 = quiver3(pos(1:n_train,1),pos(1:n_train,2),pos(1:n_train,3),mag_true(1:n_train,1)*sc,mag_true(1:n_train,2)*sc,mag_true(1:n_train,3)*sc,'r');
q2.AutoScale = 'off';
q2.LineWidth = 1.0;
scatter3(pos(1:n_train,1),pos(1:n_train,2),pos(1:n_train,3),10,'r','MarkerFaceColor','r')
hold off
axis square
grid on
xlim([-1.4 2.6])
ylim([-3 1.1])
% zlim([0 2.2])
xlabel('x (m)','Interpreter','Latex','FontSize',25)
ylabel('y (m)','Interpreter','Latex','FontSize',25)
zlabel('z (m)','Interpreter','Latex','FontSize',25)
box on
view(gca,-48.1539,27.3021)

