function mf = calculate_mf_Na2B12H12(RH)
% calculate_mf_Na2B12H12
% Calculates mass fraction of Na2B12H12 based on Relative Humidity.
% Range: 0.8562 < RH < 1.0

% Add util folder to path if robust_fzero is not found
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end

% Check RH bounds
if RH > 1
    error("RH should be 0 < RH < 1");
end
if RH < 0.8562 || RH > 1.0
    error("RH %.4f is outside valid range [%.4f, %.4f] for Na2B12H12", RH, 0.8562, 1.0);
end

% Coefficients
A_4 = -2.696;
A_3 = -1.76;
A_2 = -0.8128;
A_1 = -0.2417;
A_0 = 1.0;

% Polynomial function: RH = A0 + A1*x + ... + A4*x^4
f = @(xi) RH - A_0 - A_1.*xi - A_2.*xi.^2 - A_3.*xi.^3 - A_4.*xi.^4;

% Solve for mass fraction using robust_fzero
mf = robust_fzero(f, 0.0002, 0.2451, 0.1227);

end
