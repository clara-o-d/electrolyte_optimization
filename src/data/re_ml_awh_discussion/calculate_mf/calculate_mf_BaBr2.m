function mf = calculate_mf_BaBr2(RH)
% Add util folder to path if needed
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end
% This function calculates the mass fraction of BaBr2 as a
% function of the Relative Humidity at a temperature of 25C
if RH > 1 
    error("RH should be 0 < RH < 1")
end 
if RH < 0.7454 || RH > 0.9587 
    error("below deliquescence relative humidity or above range") 
end  
A_4 = -26.02; 
A_3 = 35.16; 
A_2 = -18.29;
A_1 = 3.914; 
A_0 = 0.6709;
f = @(xi) RH - A_0 - A_1.*xi - A_2.*xi.^2 - A_3.*xi.^3 - A_4.*xi.^4;
mf = robust_fzero(f, 0.2292, 0.5024, 0.3658);

end
% ---------------------------------------------------------
