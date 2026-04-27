function mf = calculate_mf_SrCl2(RH)
% Add util folder to path if needed
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end
% This function calculates the mass fraction of SrCl2 as a
% function of the Relative Humidity at a temperature of 25C
if RH > 1 
    error("RH should be 0 < RH < 1")
end 
if RH < 0.7235 || RH > 0.9669 
    error("below deliquescence relative humidity or above range") 
end  
A_4 = -257.1;
A_3 = 211.1;
A_2 = -63.59;
A_1 = 7.433;
A_0 = 0.6749;
f = @(xi) RH - A_0 - A_1.*xi - A_2.*xi.^2 - A_3.*xi.^3 - A_4.*xi.^4;
mf = robust_fzero(f, 0.095, 0.3368, 0.2159);

end
% ---------------------------------------------------------
