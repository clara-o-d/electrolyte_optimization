function mf = calculate_mf_HBr(RH)
% calculate_mf_HBr
% Calculates mass fraction of HBr based on Relative Humidity.
% Range: 0.9621 < RH < 0.9966
% Data source: Paper 25 in sources_list_.xlsx - Hydrobromic acid data at 25°

% Add util folder to path if robust_fzero is not found
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end

% Check RH bounds
if RH > 1
    error("RH should be 0 < RH < 1");
end
if RH < 0.9621 || RH > 0.9966
    error("RH %.4f is outside valid range [0.9621, 0.9966] for HBr", RH, 0.9437, 0.9949);
end

% Coefficients from polynomial fit: RH = A0 + A1*x + A2*x^2 + A3*x^3 + A4*x^4
A_4 = -65.535214295036;
A_3 = 6.866427960463;
A_2 = -1.559433519919;
A_1 = -0.398783529587;
A_0 = 0.999883658158;

% Polynomial function
f = @(xi) RH - A_0 - A_1.*xi - A_2.*xi.^2 - A_3.*xi.^3 - A_4.*xi.^4;

% Solve for mass fraction using robust_fzero
mf = robust_fzero(f, 0.0080, 0.0749, 0.0414);

end
