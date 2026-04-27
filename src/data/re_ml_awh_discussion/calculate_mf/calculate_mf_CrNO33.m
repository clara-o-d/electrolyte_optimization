function mf = calculate_mf_CrNO33(RH)
% calculate_mf_CrNO33
% Calculates mass fraction of Cr(NO3)3 based on Relative Humidity.
% Range: 0.8481 < RH < 0.9943
% Data source: sources_list_.xlsx - Chromium nitrate data at 25°

% Add util folder to path if robust_fzero is not found
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end

% Check RH bounds
if RH > 1
    error("RH should be 0 < RH < 1");
end
if RH < 0.8481 || RH > 0.9943
    error("RH %.4f is outside valid range [0.8481, 0.9943] for Cr(NO3)3", RH, 0.8838, 0.9957);
end

% Coefficients from polynomial fit: RH = A0 + A1*x + A2*x^2 + A3*x^3 + A4*x^4
A_4 = 2.752801968566;
A_3 = -4.571420917669;
A_2 = -0.266410008237;
A_1 = -0.241976024620;
A_0 = 1.000179106894;

% Polynomial function
f = @(xi) RH - A_0 - A_1.*xi - A_2.*xi.^2 - A_3.*xi.^3 - A_4.*xi.^4;

% Solve for mass fraction using robust_fzero
mf = robust_fzero(f, 0.0232, 0.2631, 0.1432);

end
