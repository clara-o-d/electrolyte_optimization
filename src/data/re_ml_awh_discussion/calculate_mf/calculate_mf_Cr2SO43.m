function mf = calculate_mf_Cr2SO43(RH)
% calculate_mf_Cr2SO43
% Calculates mass fraction of Cr2(SO4)3 based on Relative Humidity.
% Range: 0.8531 < RH < 0.9963
% Data source: sources_list_.xlsx - Chromium sulfate data at 25°

% Add util folder to path if robust_fzero is not found
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end

% Check RH bounds
if RH > 1
    error("RH should be 0 < RH < 1");
end
if RH < 0.8531 || RH > 0.9963
    error("RH %.4f is outside valid range [0.8531, 0.9963] for Cr2(SO4)3", RH, 0.9091, 0.9978);
end

% Coefficients from polynomial fit: RH = A0 + A1*x + A2*x^2 + A3*x^3 + A4*x^4
A_4 = -6.568363985841;
A_3 = -0.404948818532;
A_2 = 0.098514704205;
A_1 = -0.107111013599;
A_0 = 1.000290781979;

% Polynomial function
f = @(xi) RH - A_0 - A_1.*xi - A_2.*xi.^2 - A_3.*xi.^3 - A_4.*xi.^4;

% Solve for mass fraction using robust_fzero
mf = robust_fzero(f, 0.0377, 0.3544, 0.1961);

end
