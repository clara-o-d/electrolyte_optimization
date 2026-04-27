function mf = calculate_mf_HI(RH)
% calculate_mf_HI
% Calculates mass fraction of HI based on Relative Humidity.
% Range: 0.8471 < RH < 0.9966
% Data source: Paper 25 in sources_list_.xlsx - Hydroiodic acid data at 25°

% Add util folder to path if robust_fzero is not found
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end

% Check RH bounds
if RH > 1
    error("RH should be 0 < RH < 1");
end
if RH < 0.8471 || RH > 0.9966
    error("RH %.4f is outside valid range [0.8471, 0.9966] for HI", RH, 0.7797, 0.9949);
end

% Coefficients from polynomial fit: RH = A0 + A1*x + A2*x^2 + A3*x^3 + A4*x^4
A_4 = -2.824773589623;
A_3 = -0.638945352964;
A_2 = -0.657594937288;
A_1 = -0.259402999933;
A_0 = 0.999946523858;

% Polynomial function
f = @(xi) RH - A_0 - A_1.*xi - A_2.*xi.^2 - A_3.*xi.^3 - A_4.*xi.^4;

% Solve for mass fraction using robust_fzero
mf = robust_fzero(f, 0.0126, 0.2773, 0.1450);

end
