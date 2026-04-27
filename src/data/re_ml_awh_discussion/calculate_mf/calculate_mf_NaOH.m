function mf = calculate_mf_NaOH(RH)
% Add util folder to path if needed
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end
% This function calculates the mass fraction of Lithium Bromide as a
% function of the Relative Humidity at a temperature of 25C
% Fit on experimental data from: https://www.sciencedirect.com/science/article/pii/036031998590093X

if RH > 1 
    error("RH should be 0 < RH < 1")
end 
A_4 = 22.452177; 
A_3 = - 8.992202; 
A_2 =  - 3.713377;
A_1 = - 0.506960; 
A_0 = 1.003610;

f = @(X) RH - A_0 - A_1.*X - A_2.*X.^2 - A_3.*X.^3 - A_4.*X.^4;
mf = robust_fzero(f, 0.01, 0.65, 0.30);


end
