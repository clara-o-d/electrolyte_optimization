function mf = calculate_mf_Na2C4H2O4(RH)
% Add util folder to path if needed
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end
% This function calculates the mass fraction of Na2C4H2O4 as a
% function of the Relative Humidity at a temperature of 25C
if RH > 1 
    error("RH should be 0 < RH < 1")
end 
if RH < 0.8696 || RH > 0.9999 
    error("below deliquescence relative humidity or above range") 
end  
A_4 = -0.1351; 
A_3 = -1.3681; 
A_2 = -0.0731;
A_1 = -0.2492; 
A_0 = 0.9999;
f = @(xi) RH - A_0 - A_1.*xi - A_2.*xi.^2 - A_3.*xi.^3 - A_4.*xi.^4;
mf = robust_fzero(f, 0.0002, 0.3154, 0.1578);

end
% ---------------------------------------------------------
