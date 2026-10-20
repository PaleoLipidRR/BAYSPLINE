% verify_original.m — run the ORIGINAL BAYSPLINE functions and save their output.
%
% Calls UK_predict.m, UK_forward.m and checkSeasonality.m from the repository
% root, unmodified, and writes docs/BAYSPLINEpy/audit/original_matlab.mat for
% BAYSPLINEpy/tests/test_vs_original.py and notebooks/compare_implementations.ipynb.
%
% Run from the repository root, in a shell (not at the MATLAB prompt):
%
%   matlab -batch "addpath('docs/BAYSPLINEpy/audit'); verify_original"
%   octave --no-gui --quiet --eval "pkg load statistics; addpath('docs/BAYSPLINEpy/audit'); verify_original"
%
% Under Octave, the Curve Fitting Toolbox (augknt, spmak, fnxtr, fnval) and the
% Statistics Toolbox's ksdensity do not exist; this script then adds
% audit/octave_shims/, which implements them from MATLAB's documented
% behaviour. Under MATLAB the shims are never on the path. UK_predict's parfor
% runs as a plain for loop in Octave; the numbers do not depend on that.
%
% Takes ~1 minute in MATLAB with a parallel pool, ~15 minutes in Octave.

here = fileparts(mfilename('fullpath'));
is_octave = exist('OCTAVE_VERSION', 'builtin') ~= 0;
if exist('spmak') == 0 || exist('fnxtr') == 0
    addpath(fullfile(here, 'octave_shims'));
    shims = 'audit/octave_shims (augknt, spmak, fnxtr, fnval, ksdensity)';
else
    shims = 'none: Curve Fitting Toolbox present';
end
set(0, 'defaultfigurevisible', 'off');

if is_octave
    rand('twister', 2018); randn('state', 2018);
    engine = ['Octave ' version()];
else
    rng(2018, 'twister');
    engine = ['MATLAB ' version()];
end

% --- 1. DemoScript.m: Gulf of Aden, pstd = 5 -------------------------------
% readtable is MATLAB-only; dlmread skips the header row the same way.
demo_raw = dlmread('DemoUKData.csv', ',', 1, 0);
demo_input = struct('uk', demo_raw(:, 3), 'ageBP', demo_raw(:, 2), 'pstd', 5);
t0 = tic;
demo = UK_predict(demo_input.uk, demo_input.pstd);
demo_seconds = toc(t0);
close all;

% --- 2. A low-UK'37 case on the other side of the jump-width rule -----------
% brews/baysplinepy's own test case: uk = [0.4 0.5 0.6], pstd = 7.5.
low_input = struct('uk', [0.4; 0.5; 0.6], 'pstd', 7.5);
low = UK_predict(low_input.uk, low_input.pstd);
close all;

% --- 3. UK_forward.m -------------------------------------------------------
forward_input = struct('ssts', (-5:2.5:35)');
forward = UK_forward(forward_input.ssts);

% --- 4. The spline itself, with no randomness ------------------------------
bp = load('bayes_posterior_v2.mat');
spline_x = [-10; -0.4; 0; 5; 10; 15; 20; 23.6; 25; 28; 29.6; 32; 40];
spline_mean = fnval(fnxtr(spmak(augknt(bp.knots, 3), bp.bdraws)), spline_x);

% --- 5. checkSeasonality.m -------------------------------------------------
% One point inside each region, points on edges and vertices (inpolygon counts
% them as inside), the dateline for the North Pacific, and open ocean.
season_lat = [40; 60; 55; 20; 36.25; 48; 45; 30; 0; -45; 50; 60; 47.5];
season_lon = [15; -20; -170; -40; -5.5; -30; 180; 20; 0; 170; 175; -4.5; 3];
months = checkSeasonality(season_lat, season_lon);
season_months = false(numel(months), 12);       % cell arrays don't round-trip; a mask does
for i = 1:numel(months)
    season_months(i, months{i}) = true;
end

provenance = struct('engine', engine, 'shims', shims, ...
                    'when', datestr(now, 'yyyy-mm-dd HH:MM:SS'), ...
                    'seed', 2018, 'demo_seconds', demo_seconds);

out = fullfile(here, 'original_matlab.mat');
save('-v7', out, 'demo_input', 'demo', 'low_input', 'low', 'forward_input', ...
     'forward', 'spline_x', 'spline_mean', 'season_lat', 'season_lon', ...
     'season_months', 'provenance');
fprintf('wrote %s (%s; shims: %s)\n', out, engine, shims);
