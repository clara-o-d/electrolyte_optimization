function mf = calculate_mf_LiBr(RH)
% Add util folder to path if needed
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end
% This function calculates the mass fraction of Lithium Bromide as a
% function of the Relative Humidity at a temperature of 25C
% Fit on experimental data from: https://doi.org/10.1002/er.1790

if RH > 1 
    error("RH should be 0 < RH < 1")
end 
A_4 = 18.838641746059196; 
A_3 = -19.878321345922046; 
A_2 = 3.871666208430489;
A_1 = -0.766050042576149; 
A_0 = 1.004948395640807;

f = @(X) RH - A_0 - A_1.*X - A_2.*X.^2 - A_3.*X.^3 - A_4.*X.^4;
mf = robust_fzero(f, 0.01, 0.65, 0.25);


end
