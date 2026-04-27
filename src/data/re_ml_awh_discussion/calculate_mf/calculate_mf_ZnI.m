function mf = calculate_mf_ZnI(RH)
% Add util folder to path if needed
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end
% This function calculates the mass fraction of Zinc Iodide as a
% function of the Relative Humidity at a temperature of 25C
% Fit on pressure data from: https://srd.nist.gov/jpcrdreprint/1.555639.pdf

if RH > 1 
    error("RH should be 0 < RH < 1")
end 
A_4 = 7.750630; 
A_3 = - 24.365130; 
A_2 = 22.380262;
A_1 = -8.657558; 
A_0 = 2.104603;

f = @(xi) RH - A_0 - A_1.*xi - A_2.*xi.^2 - A_3.*xi.^3 - A_4.*xi.^4;
mf = robust_fzero(f, 0.01, 0.9, 0.30);

end
