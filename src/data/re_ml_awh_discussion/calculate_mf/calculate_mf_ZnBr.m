function mf = calculate_mf_ZnBr(RH)
% Add util folder to path if needed
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end
% This function calculates the mass fraction of Zinc Bromide as a
% function of the Relative Humidity at a temperature of 25C
% Function coming from isopiestic data (H2SO4) from DOI	https://doi.org/10.1039/TF9403600733

if RH > 0.87 
    error("RH should be 0 < RH < 0.87")
end 
MM = 225.198;                     % [g/mol] ZnCl2

A_4 = -2.441539790429647e-05;
A_3 = 0.001164035024612; 
A_2 = -0.016402311078937; 
A_1 = 0.015286715512812;
A_0 = 0.921783882286604; 

f = @(molality) RH - A_0 - A_1.*molality - A_2.*molality.^2 - A_3.*molality.^3 - A_4.*molality.^4;
m = robust_fzero(f, 0.01, 30, 5);
mf = (m.*MM)./(1000 + m.*MM); % mass fraction

end
