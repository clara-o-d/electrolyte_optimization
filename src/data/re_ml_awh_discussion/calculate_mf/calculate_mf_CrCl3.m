function mf = calculate_mf_CrCl3(RH)
% calculate_mf_CrCl3
% Calculates mass fraction of CrCl3 based on Relative Humidity.
% Range: 0.8548 < RH < 0.9941
% Data source: sources_list_.xlsx - Chromium chloride data at 25°

% Add util folder to path if robust_fzero is not found
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end

% Check RH bounds
if RH > 1
    error("RH should be 0 < RH < 1");
end
if RH < 0.8548 || RH > 0.9941
    error("RH %.4f is outside valid range [0.8548, 0.9941] for CrCl3", RH, 0.8890, 0.9956);
end

% Coefficients from polynomial fit: RH = A0 + A1*x + A2*x^2 + A3*x^3 + A4*x^4
A_4 = 59.705833699635;
A_3 = -24.004113839832;
A_2 = 0.051088725312;
A_1 = -0.376154083644;
A_0 = 1.000101936564;

% Polynomial function
f = @(xi) RH - A_0 - A_1.*xi - A_2.*xi.^2 - A_3.*xi.^3 - A_4.*xi.^4;

% Solve for mass fraction using robust_fzero
mf = robust_fzero(f, 0.0156, 0.1815, 0.0985);

end
