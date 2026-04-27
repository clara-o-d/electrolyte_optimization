function mf = calculate_mf_ZnCl(RH)
% Add util folder to path if needed
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end
% This function calculates the mass fraction of Zinc Chloride as a
% function of the Relative Humidity at a temperature of 60C
% Fit on pressure data from: https://pubs.acs.org/doi/pdf/10.1021/ic50195a058

if RH > 1 
    error("RH should be 0 < RH < 1")
end 
A_4 = 13.389423002312885; 
A_3 = -19.289870089311886; 
A_2 = 6.562516975212110;
A_1 = -0.984267594678104; 
A_0 = 1.009182113918304;

f = @(xi) RH - A_0 - A_1.*xi - A_2.*xi.^2 - A_3.*xi.^3 - A_4.*xi.^4;
mf = robust_fzero(f, 0.01, 0.8, 0.40);

end
