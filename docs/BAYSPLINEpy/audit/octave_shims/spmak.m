function sp = spmak(knots, coefs)
% Octave stand-in for the Curve Fitting Toolbox spmak: the B-form of a spline.
% coefs is d-by-n (one row per component), as in MATLAB; order = #knots - n.
% Loaded by verify_original.m ONLY when the real function is absent.
[d, n] = size(coefs);
sp = struct('form', 'B-', 'knots', knots(:).', 'coefs', coefs, ...
            'number', n, 'order', numel(knots) - n, 'dim', d, 'xtr', false);
end
