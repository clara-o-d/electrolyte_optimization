function mf = calculate_mf_NH4Br(RH)
% calculate_mf_NH4Br
% Calculates mass fraction of NH4Br based on Relative Humidity.
% Range: 0.808 < RH < 0.9967

% Add util folder to path if robust_fzero is not found
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end

% Check RH bounds
if RH > 1
    error("RH should be 0 < RH < 1");
end
if RH < 0.808 || RH > 0.9967
    error("RH %.4f is outside valid range [%.4f, %.4f] for NH4Br", RH, 0.808, 0.9967);
end

% Coefficients
A_4 = -0.09239;
A_3 = -0.4958;
A_2 = -0.3155;
A_1 = -0.3289;
A_0 = 0.9999;

% Polynomial function: RH = A0 + A1*x + ... + A4*x^4
f = @(xi) RH - A_0 - A_1.*xi - A_2.*xi.^2 - A_3.*xi.^3 - A_4.*xi.^4;

% Solve for mass fraction using robust_fzero
mf = robust_fzero(f, 0.0097, 0.3701, 0.1899);

end
