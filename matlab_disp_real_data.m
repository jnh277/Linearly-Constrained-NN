clear all
clc

load('real_data/magnetic_field_data')

fx = scatteredInterpolant(pos(:,1),pos(:,2),pos(:,3),mag(:,1));
fy = scatteredInterpolant(pos(:,1),pos(:,2),pos(:,3),mag(:,2));
fz = scatteredInterpolant(pos(:,1),pos(:,2),pos(:,3),mag(:,3));

[X, Y, Z] = meshgrid(linspace(min(pos(:,1)),max(pos(:,1)),5),linspace(min(pos(:,2)),max(pos(:,2)),5),...
    linspace(min(pos(:,3)),max(pos(:,3)),5));

mag_x = fx(X,Y,Z);
mag_y = fy(X,Y,Z);
mag_z = fz(X,Y,Z);


figure(1)
clf
quiver3(X(:),Y(:),Z(:),mag_x(:),mag_y(:),mag_z(:))

figure(2)
quiver3(pos(:,1),pos(:,2),pos(:,3),mag(:,1),mag(:,2),mag(:,3))

ind = logical((pos(:,2) > 0.5-0.01).*(pos(:,2) < 0.5+0.01));

figure(3)
quiver(pos(ind,1),pos(ind,3),mag(ind,1),mag(ind,3))