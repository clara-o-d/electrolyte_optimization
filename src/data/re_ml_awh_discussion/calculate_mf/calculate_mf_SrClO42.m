function mf = calculate_mf_SrClO42(RH)
% calculate_mf_SrClO42
% Calculates mass fraction of Sr(ClO4)2 based on Relative Humidity.
% Range: 0.2393 < RH < 0.9953
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
if RH < 0.2393 || RH > 0.9953
    error("RH %.4f is outside valid range [%.4f, %.4f] for Sr(ClO4)2", RH, 0.2393, 0.9953);
end

% Coefficients from polynomial fit: RH = A0 + A1*x + A2*x^2 + A3*x^3 + A4*x^4
A_4 = 4.280413721605415;
A_3 = -7.514681064213954;
A_2 = 2.265191337651363;
A_1 = -0.497600929019958;
A_0 = 1.011728293309786;

% Polynomial function
f = @(xi) RH - A_0 - A_1.*xi - A_2.*xi.^2 - A_3.*xi.^3 - A_4.*xi.^4;

% Solve for mass fraction using robust_fzero
mf = robust_fzero(f, 0.0279, 0.6962, 0.3621);

end
