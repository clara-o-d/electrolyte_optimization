function mf = calculate_mf_MgSO4(RH)
% Add util folder to path if needed
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end
% This function calculates the mass fraction of MgSO4 as a
% function of the Relative Humidity at a temperature of 25C
if RH > 1 
    error("RH should be 0 < RH < 1")
end 
if RH < 0.905 || RH > 0.996 
    error("below deliquescence relative humidity or above range") 
end  
A_4 = -8.77; 
A_3 = -0.2989; 
A_2 = -0.09264;
A_1 = -0.1462; 
A_0 = 0.9994;
f = @(xi) RH - A_0 - A_1.*xi - A_2.*xi.^2 - A_3.*xi.^3 - A_4.*xi.^4;
mf = robust_fzero(f, 0.0235, 0.2653, 0.1444);

end
% ---------------------------------------------------------
