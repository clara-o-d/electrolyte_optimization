function mf = calculate_mf_LiI(RH)
% Add util folder to path if needed
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end
% This function calculates the mass fraction of Lithium Iodide as a
% function of the Relative Humidity at a temperature of 30C
% Fit on vapour pressure data at 30C from: https://pubs.acs.org/doi/pdf/10.1021/je00060a020

if RH > 1 
    error("RH should be 0 < RH < 1")
end 
A_5 = -1.121229305663804;
A_4 = -6.616780479521403; 
A_3 = 4.951108742483354; 
A_2 = -2.2094102170;
A_1 = -0.16556692518410; 
A_0 = 1.000055934279976;

f = @(xi) RH - A_0 - A_1.*xi - A_2.*xi.^2 - A_3.*xi.^3 - A_4.*xi.^4 - A_5.*xi.^5;
mf = robust_fzero(f, 0.01, 0.62, 0.25);

if mf > 0.62
    error("below deliquescence relative humidity")
end 

end
