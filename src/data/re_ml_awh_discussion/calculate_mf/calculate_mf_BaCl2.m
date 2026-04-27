function mf = calculate_mf_BaCl2(RH)
% Add util folder to path if needed
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end
% This function calculates the mass fraction of BaCl2 as a
% function of the Relative Humidity at a temperature of 25C
if RH > 1 
    error("RH should be 0 < RH < 1")
end 
if RH < 0.9078 || RH > 0.9600 
    error("below deliquescence relative humidity or above range") 
end  
A_4 = 470.9;
A_3 = -302.2;
A_2 = 68.64;
A_1 = -6.871;
A_0 = 1.214;
f = @(xi) RH - A_0 - A_1.*xi - A_2.*xi.^2 - A_3.*xi.^3 - A_4.*xi.^4;
mf = robust_fzero(f, 0.095, 0.2242, 0.1596);

end
% ---------------------------------------------------------
