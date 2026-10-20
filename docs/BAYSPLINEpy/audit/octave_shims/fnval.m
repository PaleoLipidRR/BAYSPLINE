function v = fnval(f, x)
% Octave stand-in for the Curve Fitting Toolbox fnval on a B-form spline.
% Evaluates by the Cox-de Boor recursion; at the right end of the basic interval
% it takes the limit from the left, as fnval does. If the spline went through
% fnxtr, points outside the interval follow the tangent line at the nearer end.
% Output: size(x) for a scalar spline, d-by-numel(x) for a d-valued one.
% Loaded by verify_original.m ONLY when the real function is absent.
t = f.knots; k = f.order; c = f.coefs;
a = t(k); b = t(end - k + 1);
xs = x(:);
inside = xs >= a & xs <= b;
xc = min(max(xs, a), b);
v = c * basis(t, k, xc).';                     % d-by-numel(x)
lo = find(xs < a); hi = find(xs > b);
if f.xtr
    if ~isempty(lo) || ~isempty(hi)
        % derivative: order k-1 spline, coefficients (k-1)(c_{i+1}-c_i)/(t_{i+k}-t_{i+1})
        n = size(c, 2);
        dc = (k - 1) * (c(:, 2:n) - c(:, 1:n-1)) ./ (t(k+1:n+k-1) - t(2:n));
        dt = t(2:end-1);
        ends = basis(t, k, [a; b]).';   dends = basis(dt, k - 1, [a; b]).';
        va = c * ends(:, 1);   vb = c * ends(:, 2);
        da = dc * dends(:, 1); db = dc * dends(:, 2);
        v(:, lo) = va + da * (xs(lo).' - a);
        v(:, hi) = vb + db * (xs(hi).' - b);
    end
else
    v(:, ~inside) = NaN;
end
if size(c, 1) == 1
    v = reshape(v, size(x));
end
end

function B = basis(t, k, x)
% numel(x)-by-n matrix of order-k B-splines on knots t, by Cox-de Boor.
x = x(:); t = t(:).';
m = numel(t) - 1;
B = double(x >= t(1:m) & x < t(2:m+1));
last = find(t < t(end), 1, 'last');            % the last nonempty interval
atend = x == t(end);
B(atend, :) = 0;
B(atend, last) = 1;                            % left limit at the right end
for r = 2:k
    i = 1:(m - r + 1);
    d1 = t(i + r - 1) - t(i);  d2 = t(i + r) - t(i + 1);
    w1 = (x - t(i)) ./ d1;     w1(:, d1 == 0) = 0;
    w2 = (t(i + r) - x) ./ d2; w2(:, d2 == 0) = 0;
    B = w1 .* B(:, i) + w2 .* B(:, i + 1);
end
end
