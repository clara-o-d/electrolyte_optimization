function mf = calculate_mf_CaI2(RH)
% Add util folder to path if needed
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end
% This function calculates the mass fraction of CaI2 as a
% function of the Relative Humidity at a temperature of 25C
if RH > 1 
    error("RH should be 0 < RH < 1")
end 
if RH < 0.7590 || RH > 0.9294 
    error("below deliquescence relative humidity or above range") 
end  
A_4 = -323.7;
A_3 = 439.4;
A_2 = -221.8;
A_1 = 48.72;
A_0 = -2.989;
f = @(xi) RH - A_0 - A_1.*xi - A_2.*xi.^2 - A_3.*xi.^3 - A_4.*xi.^4;
mf = robust_fzero(f, 0.2506, 0.4614, 0.356);

end
% ---------------------------------------------------------
