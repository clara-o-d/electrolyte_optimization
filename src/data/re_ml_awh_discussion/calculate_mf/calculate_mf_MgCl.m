function mf = calculate_mf_MgCl(RH)
% Add util folder to path if needed
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end
% This function calculates the mass fraction of Magnesium Cloride as a
% function of the Relative Humidity at a temperature of 20-24C
% Fit on data from: https://doi.org/10.1080/027868299304219

if RH > 1 
    error("RH should be 0 < RH < 1")
end 
A_4 = 186.32487108;
A_3 = -153.67496570; 
A_2 = 38.21982328; 
A_1 = -4.86704441;
A_0 = 1.16231287; 

f = @(X) RH - A_0 - A_1.*X - A_2.*X.^2 - A_3.*X.^3 - A_4.*X.^4;
mf = robust_fzero(f, 0.01, 0.75, 0.30);

end