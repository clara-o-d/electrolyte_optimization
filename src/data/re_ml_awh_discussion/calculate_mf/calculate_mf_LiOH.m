function mf = calculate_mf_LiBr(RH)
% Add util folder to path if needed
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end
% This function calculates the mass fraction of Lithium Bromide as a
% function of the Relative Humidity at a temperature of 25C
% Fit on experimental data from: https://pubs.acs.org/doi/epdf/10.1021/ie0489148

if RH > 1 
    error("RH should be 0 < RH < 1")
end 
A_4 = 0; 
A_3 = 0; 
A_2 = -0.8825;
A_1 = -1.3171; 
A_0 = 0.9993;

f = @(X) RH - A_0 - A_1.*X - A_2.*X.^2 - A_3.*X.^3 - A_4.*X.^4;
mf = robust_fzero(f, 0.01, 0.65, 0.25);


end
