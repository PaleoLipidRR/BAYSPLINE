function [f, xi] = ksdensity(x, xi)
% Octave stand-in for the Statistics Toolbox ksdensity, used by UK_predict.m
% only to draw its prior-vs-posterior figure. A Gaussian kernel with Silverman's
% bandwidth; nothing it returns reaches UK_predict's output structure.
% Loaded by verify_original.m ONLY when the real function is absent.
x = x(:); xi = xi(:);
h = 1.06 * std(x) * numel(x)^(-1/5);
f = zeros(size(xi));
for j = 1:numel(xi)
    f(j) = mean(exp(-0.5 * ((xi(j) - x) / h).^2)) / (h * sqrt(2 * pi));
end
end
