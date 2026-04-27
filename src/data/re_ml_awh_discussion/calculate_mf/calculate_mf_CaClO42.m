function mf = calculate_mf_CaClO42(RH)
% calculate_mf_CaClO42
% Calculates mass fraction of Ca(ClO4)2 based on Relative Humidity.
% Range: 0.2211 < RH < 0.9952
% Data source: Paper 23 - The Osmotic and Activity Coefficients of Calcium, 
% Strontium and Barium Perchlorate at 25°

% Add util folder to path if robust_fzero is not found
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end

% Check RH bounds
if RH > 1
    error("RH should be 0 < RH < 1");
end
if RH < 0.2211 || RH > 0.9952
    error("RH %.4f is outside valid range [%.4f, %.4f] for Ca(ClO4)2", RH, 0.2211, 0.9952);
end

% Coefficients from polynomial fit: RH = A0 + A1*x + A2*x^2 + A3*x^3 + A4*x^4
A_4 = 7.603133858184870;
A_3 = -10.920520566456585;
A_2 = 2.695439231592639;
A_1 = -0.542031480000002;
A_0 = 1.010171694686977;

% Polynomial function
f = @(xi) RH - A_0 - A_1.*xi - A_2.*xi.^2 - A_3.*xi.^3 - A_4.*xi.^4;

% Solve for mass fraction using robust_fzero
mf = robust_fzero(f, 0.0233, 0.6259, 0.3246);

end
