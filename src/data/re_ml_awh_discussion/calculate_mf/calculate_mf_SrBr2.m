function mf = calculate_mf_SrBr2(RH)
% Add util folder to path if needed
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end
% This function calculates the mass fraction of SrBr2 as a
% function of the Relative Humidity at a temperature of 25C
if RH > 1 
    error("RH should be 0 < RH < 1")
end 
if RH < 0.6857 || RH > 0.9364 
    error("below deliquescence relative humidity or above range") 
end  
A_4 = -11.94;
A_3 = 14.8;
A_2 = -9.595;
A_1 = 2.226;
A_0 = 0.7723;
f = @(xi) RH - A_0 - A_1.*xi - A_2.*xi.^2 - A_3.*xi.^3 - A_4.*xi.^4;
mf = robust_fzero(f, 0.1652, 0.4525, 0.3089);

end
% ---------------------------------------------------------
