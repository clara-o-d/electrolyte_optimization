function mf = calculate_mf_K2SO4(RH)
% Add util folder to path if needed
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end
% This function calculates the mass fraction of K2SO4 as a
% function of the Relative Humidity at a temperature of 25C
if RH > 1 
    error("RH should be 0 < RH < 1")
end 
if RH < 0.972 || RH > 0.9958 
    error("below deliquescence relative humidity or above range") 
end  
A_4 = 36.35; 
A_3 = -10.29; 
A_2 = 0.981;
A_1 = -0.2619; 
A_0 = 1.0;
f = @(xi) RH - A_0 - A_1.*xi - A_2.*xi.^2 - A_3.*xi.^3 - A_4.*xi.^4;
mf = robust_fzero(f, 0.0171, 0.1224, 0.0697);

end
% ---------------------------------------------------------
