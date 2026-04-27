function mf = calculate_mf_NaF(RH)
% calculate_mf_NaF
% Calculates mass fraction of NaF based on Relative Humidity.
% Range: 0.9691 < RH < 0.9967
% Data source: Paper 25 in sources_list_.xlsx - Sodium fluoride data at 25°

% Add util folder to path if robust_fzero is not found
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end

% Check RH bounds
if RH > 1
    error("RH should be 0 < RH < 1");
end
if RH < 0.9691 || RH > 0.9967
    error("RH %.4f is outside valid range [0.9691, 0.9967] for NaF", RH, 0.9540, 0.9950);
end

% Coefficients from polynomial fit: RH = A0 + A1*x + A2*x^2 + A3*x^3 + A4*x^4
A_4 = 374.945938934898;
A_3 = -45.346423325845;
A_2 = 1.735565644980;
A_1 = -0.786807084335;
A_0 = 0.999938032657;

% Polynomial function
f = @(xi) RH - A_0 - A_1.*xi - A_2.*xi.^2 - A_3.*xi.^3 - A_4.*xi.^4;

% Solve for mass fraction using robust_fzero
mf = robust_fzero(f, 0.0042, 0.0403, 0.0222);

end
