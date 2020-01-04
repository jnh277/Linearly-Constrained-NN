syms g11 g12 g13 g21 g22 g23 g31 g32 g33
syms dx dy dz nu

C = [dx, nu*dx,(1-nu)*dy;
    nu*dy, dy, (1-nu)*dx];
G = [g11 g12 g13;
    g21 g22 g23;
    g31 g32 g33];
Q = [dx*dx;dy*dy;dx*dy];
F = C*G*Q;