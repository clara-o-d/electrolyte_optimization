function mf = calculate_mf_NH4Cl(RH)
% Add util folder to path if needed
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end
% This function calculates the mass fraction of NH4Cl as a
% function of the Relative Humidity at a temperature of 25C
if RH > 1 
    error("RH should be 0 < RH < 1")
end 
if RH < 0.812 || RH > 0.993 
    error("below deliquescence relative humidity or above range") 
end  
A_4 = 22.48; 
A_3 = -10.93; 
A_2 = 0.8213;
A_1 = -0.6539; 
A_0 = 1.0;
f = @(xi) RH - A_0 - A_1.*xi - A_2.*xi.^2 - A_3.*xi.^3 - A_4.*xi.^4;
mf = robust_fzero(f, 0.0106, 0.243, 0.1268);

end
% ---------------------------------------------------------
