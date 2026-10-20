function augknot = augknt(knots, k)
% Octave stand-in for the Curve Fitting Toolbox augknt (two-argument form).
% Loaded by verify_original.m ONLY when the real function is absent.
% Breaks are made unique; the end knots get multiplicity exactly k.
knots = unique(knots(:).');
augknot = [repmat(knots(1), 1, k), knots(2:end-1), repmat(knots(end), 1, k)];
end
