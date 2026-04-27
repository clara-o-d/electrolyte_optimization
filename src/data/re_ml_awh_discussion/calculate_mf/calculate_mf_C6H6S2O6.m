function mf = calculate_mf_C6H6S2O6(RH)
% calculate_mf_C6H6S2O6
% Calculates mass fraction of C6H6S2O6 based on Relative Humidity.
% Range: 0.8697 < RH < 1.0

% Add util folder to path if robust_fzero is not found
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end

% Check RH bounds
if RH > 1
    error("RH should be 0 < RH < 1");
end
if RH < 0.8697 || RH > 1.0
    error("RH %.4f is outside valid range [%.4f, %.4f] for C6H6S2O6", RH, 0.8697, 1.0);
end

% Coefficients
A_4 = -2.415;
A_3 = -0.4434;
A_2 = -0.4526;
A_1 = -0.1933;
A_0 = 1.0;

% Polynomial function: RH = A0 + A1*x + ... + A4*x^4
f = @(xi) RH - A_0 - A_1.*xi - A_2.*xi.^2 - A_3.*xi.^3 - A_4.*xi.^4;

% Solve for mass fraction using robust_fzero
mf = robust_fzero(f, 0.0002, 0.3001, 0.1501);

end
