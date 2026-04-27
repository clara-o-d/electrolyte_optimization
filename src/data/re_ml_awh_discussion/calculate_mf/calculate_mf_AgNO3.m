function mf = calculate_mf_AgNO3(RH)
% Add util folder to path if needed
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end
% This function calculates the mass fraction of AgNO3 as a
% function of the Relative Humidity at a temperature of 25C
if RH > 1 
    error("RH should be 0 < RH < 1")
end 
if RH < 0.855 || RH > 0.986 
    error("below deliquescence relative humidity or above range") 
end  
A_4 = -0.5541; 
A_3 = 0.4497; 
A_2 = -0.1387;
A_1 = -0.1572; 
A_0 = 0.999;
f = @(xi) RH - A_0 - A_1.*xi - A_2.*xi.^2 - A_3.*xi.^3 - A_4.*xi.^4;
mf = robust_fzero(f, 0.0781, 0.6709, 0.3745);

end
% ---------------------------------------------------------
