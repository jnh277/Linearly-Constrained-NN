function [S] = skew7(v)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

S = [0,     -v(4),  -v(7),     v(2),   -v(6),   v(5),    v(3);
    v(4),   0,      -v(5),    -v(1),   v(3),    -v(7),  v(6);
    v(7),   v(5),   0,        -v(6),    -v(2),  v(4),   -v(1);
    -v(2),  v(1),   v(6),     0,        -v(7),  -v(3),  v(5);
    v(6),   -v(3),  v(2),     v(7),     0,      -v(1),  -v(4);
    -v(5),  v(7),   -v(4),    v(3),     v(1),   0,      -v(2);
    -v(3),  -v(6),  v(1),     -v(5),    v(4),  v(2),    0];


end

