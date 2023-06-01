// Gmsh project created on Wed May 31 22:30:21 2023
//+
Point(1) = {0, 0, -0, 1.0};
//+
Point(2) = {0, 1, 0, 1.0};
//+
Point(3) = {0, 2, 0, 1.0};
//+
Point(4) = {1, 2, 0, 1.0};
//+
Point(5) = {1, 1, 0, 1.0};
//+
Point(6) = {2, 1, 0, 1.0};
//+
Point(7) = {2, 0, 0, 1.0};
//+
Point(8) = {1, 0, -0, 1.0};

//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 5};
//+
Line(5) = {5, 6};
//+
Line(6) = {6, 7};
//+
Line(7) = {7, 8};
//+
Line(8) = {8, 1};
//+
Line(9) = {5, 8};

//+
Curve Loop(1) = {3, 4, 9, 8, 1, 2};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {5, 6, 7, -9};
//+
Plane Surface(2) = {2};
//+
Transfinite Surface {1} = {3, 4, 8, 1};
//+
Transfinite Surface {2} = {5, 6, 7, 8};
//+
Transfinite Curve {2, 1, 3, 4, 9, 8, 5, 6, 7} = 10 Using Progression 1;
