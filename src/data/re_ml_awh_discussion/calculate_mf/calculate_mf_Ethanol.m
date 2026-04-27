function mf = calculate_mf_Ethanol(RH)
% Add util folder to path if needed
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end
% This function calculates the mass fraction of Ethanol (non-electrolyte) as a
% function of the Relative Humidity at a temperature of 25C
if RH > 1 
    error("RH should be 0 < RH < 1")
end 
if RH < 0.0630 || RH > 0.9640 
    error("below minimum relative humidity or above range") 
end  
A_4 = -7.787275; 
A_3 = 12.797212; 
A_2 = -6.931222;
A_1 = 1.051674; 
A_0 = 0.905611;
f = @(xi) RH - A_0 - A_1.*xi - A_2.*xi.^2 - A_3.*xi.^3 - A_4.*xi.^4;
mf = robust_fzero(f, 0.073295, 0.988051, 0.530673);

end
% ---------------------------------------------------------
