function mf = calculate_mf_Na2C6H4S2O6(RH)
% Add util folder to path if needed
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end
% This function calculates the mass fraction of Na2C6H4S2O6 as a
% function of the Relative Humidity at a temperature of 25C
if RH > 1 
    error("RH should be 0 < RH < 1")
end 
if RH < 0.8280 || RH > 0.9999 
    error("below deliquescence relative humidity or above range") 
end  
A_4 = -0.5695; 
A_3 = -0.3632; 
A_2 = -0.1757;
A_1 = -0.1635; 
A_0 = 1.0000;
f = @(xi) RH - A_0 - A_1.*xi - A_2.*xi.^2 - A_3.*xi.^3 - A_4.*xi.^4;
mf = robust_fzero(f, 0.0003, 0.4585, 0.2294);

end
% ---------------------------------------------------------
