function mf = calculate_mf_NaNO3(RH)
% Add util folder to path if needed
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end
% This function calculates the mass fraction of NaNO3 as a
% function of the Relative Humidity at a temperature of 25C
if RH > 1 
    error("RH should be 0 < RH < 1")
end 
if RH < 0.9701 || RH > 0.9996 
    error("below deliquescence relative humidity or above range") 
end  
A_4 = 45.18; 
A_3 = -10.12; 
A_2 = 0.6589;
A_1 = -0.3914; 
A_0 = 0.9999;
f = @(xi) RH - A_0 - A_1.*xi - A_2.*xi.^2 - A_3.*xi.^3 - A_4.*xi.^4;
mf = robust_fzero(f, 0.0009, 0.0785, 0.0397);

end
% ---------------------------------------------------------
