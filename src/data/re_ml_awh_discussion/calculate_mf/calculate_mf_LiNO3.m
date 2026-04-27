function mf = calculate_mf_LiNO3(RH)
% Add util folder to path if needed
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end
% This function calculates the mass fraction of LiNO3 as a
% function of the Relative Humidity at a temperature of 25C
if RH > 1 
    error("RH should be 0 < RH < 1")
end 
if RH < 0.7353 || RH > 0.9967 
    error("below deliquescence relative humidity or above range") 
end  
A_4 = 8.365; 
A_3 = -6.248; 
A_2 = -0.1888;
A_1 = -0.527; 
A_0 = 1.001;
f = @(xi) RH - A_0 - A_1.*xi - A_2.*xi.^2 - A_3.*xi.^3 - A_4.*xi.^4;
mf = robust_fzero(f, 0.0068, 0.2927, 0.1497);

end
% ---------------------------------------------------------
