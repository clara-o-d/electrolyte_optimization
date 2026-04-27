function mf = calculate_mf_NH4CrSO42(RH)
% calculate_mf_NH4CrSO42
% Calculates mass fraction of NH4Cr(SO4)2 based on Relative Humidity.
% Range: 0.9757 < RH < 0.9961
% Data source: sources_list_.xlsx - Ammonium chromium sulfate data at 25°

% Add util folder to path if robust_fzero is not found
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end

% Check RH bounds
if RH > 1
    error("RH should be 0 < RH < 1");
end
if RH < 0.9757 || RH > 0.9961
    error("RH %.4f is outside valid range [0.9757, 0.9961] for NH4Cr(SO4)2", RH, 0.9817, 0.9970);
end

% Coefficients from polynomial fit: RH = A0 + A1*x + A2*x^2 + A3*x^3 + A4*x^4
A_4 = 1.082173099946;
A_3 = -1.563120148379;
A_2 = 0.139205043211;
A_1 = -0.142444374027;
A_0 = 0.999625842546;

% Polynomial function
f = @(xi) RH - A_0 - A_1.*xi - A_2.*xi.^2 - A_3.*xi.^3 - A_4.*xi.^4;

% Solve for mass fraction using robust_fzero
mf = robust_fzero(f, 0.0255, 0.1551, 0.0903);

end
