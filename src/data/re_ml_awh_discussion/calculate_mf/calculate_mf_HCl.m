function mf = calculate_mf_HCl(RH)
% Add util folder to path if needed
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end
% This function calculates the mass fraction of Magnesium Cloride as a
% function of the Relative Humidity at a temperature 25C
% Fit on data from: https://doi.org/10.1063/1.3253108

if RH > 1 
    error("RH should be 0 < RH < 1")
end 
A_4 = 0;
A_3 = 14.704; 
A_2 = -10.183; 
A_1 = -0.4859;
A_0 = 0.9965; 

f = @(X) RH - A_0 - A_1.*X - A_2.*X.^2 - A_3.*X.^3 - A_4.*X.^4;
mf = robust_fzero(f, 0.01, 0.75, 0.30);

end