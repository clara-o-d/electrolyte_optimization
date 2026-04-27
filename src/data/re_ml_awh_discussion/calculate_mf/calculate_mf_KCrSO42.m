function mf = calculate_mf_KCrSO42(RH)
% calculate_mf_KCrSO42
% Calculates mass fraction of KCr(SO4)2 based on Relative Humidity.
% Range: 0.9663 < RH < 0.9962
% Data source: sources_list_.xlsx - Potassium chromium sulfate data at 25°

% Add util folder to path if robust_fzero is not found
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end

% Check RH bounds
if RH > 1
    error("RH should be 0 < RH < 1");
end
if RH < 0.9663 || RH > 0.9962
    error("RH %.4f is outside valid range [0.9663, 0.9962] for KCr(SO4)2", RH, 0.9746, 0.9972);
end

% Coefficients from polynomial fit: RH = A0 + A1*x + A2*x^2 + A3*x^3 + A4*x^4
A_4 = 2.476456440001;
A_3 = -2.204509491668;
A_2 = 0.352786565013;
A_1 = -0.150545872420;
A_0 = 1.000148325738;

% Polynomial function
f = @(xi) RH - A_0 - A_1.*xi - A_2.*xi.^2 - A_3.*xi.^3 - A_4.*xi.^4;

% Solve for mass fraction using robust_fzero
mf = robust_fzero(f, 0.0275, 0.2207, 0.1241);

end
