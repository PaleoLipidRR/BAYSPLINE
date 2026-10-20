function f = fnxtr(f, order)
% Octave stand-in for the Curve Fitting Toolbox fnxtr. With the default order
% 2, beyond the basic interval the spline continues as the straight line that
% matches its value and first derivative at the end. Only order 2 is supported.
% Loaded by verify_original.m ONLY when the real function is absent.
if nargin > 1 && order ~= 2
    error('fnxtr shim: only the default order 2 is implemented');
end
f.xtr = true;
end
