function mf = calculate_mf_KI(RH)
% Add util folder to path if needed
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end
% This function calculates the mass fraction of KI as a
% function of the Relative Humidity at a temperature of 25C
if RH > 1 
    error("RH should be 0 < RH < 1")
end 
if RH < 0.9671 || RH > 1.0 
    error("below deliquescence relative humidity or above range") 
end  
A_4 = 5.922; 
A_3 = -2.29; 
A_2 = 0.02606;
A_1 = -0.2054; 
A_0 = 1.0;
f = @(xi) RH - A_0 - A_1.*xi - A_2.*xi.^2 - A_3.*xi.^3 - A_4.*xi.^4;
mf = robust_fzero(f, 0.0002, 0.1424, 0.0713);

end
% ---------------------------------------------------------
