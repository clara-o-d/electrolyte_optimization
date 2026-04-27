function mf = calculate_mf_Li2C6H4S2O6(RH)
% Add util folder to path if needed
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end
% This function calculates the mass fraction of Li2C6H4S2O6 as a
% function of the Relative Humidity at a temperature of 25C
if RH > 1 
    error("RH should be 0 < RH < 1")
end 
if RH < 0.8150 || RH > 0.9999 
    error("below deliquescence relative humidity or above range") 
end  
A_4 = -0.7427; 
A_3 = -1.0093; 
A_2 = -0.6419;
A_1 = -0.2378; 
A_0 = 1.0000;
f = @(xi) RH - A_0 - A_1.*xi - A_2.*xi.^2 - A_3.*xi.^3 - A_4.*xi.^4;
mf = robust_fzero(f, 0.0002, 0.3221, 0.1611);

end
% ---------------------------------------------------------
