% syms g11 g12 g13 g21 g22 g23 g31 g32 g33
% syms dx dy dz
% 
% C = [dx, dy, dz];
% G = [g11 g12 g13;
%     g21 g22 g23;
%     g31 g32 g33];
% Q = [dx;dy;dz];
% F = C*G*Q;
% 
% % S = solve(F,g11,g12,g13, g21, g22, g23, g31, g32, g33)
% 
% Cx = [dy+dz;-dx+dz;-dx-dy];
dims = 7;
dx = sym('dx',[dims, 1]);
G = sym('g',[dims,dims]);
C = dx.';

F = C*G*dx;

Gf = triu(ones(dims,dims),1) - triu(ones(dims,dims),1).';
Gx = Gf*dx;

