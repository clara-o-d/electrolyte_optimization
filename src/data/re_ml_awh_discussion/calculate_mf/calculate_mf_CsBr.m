function mf = calculate_mf_CsBr(RH)
% Add util folder to path if needed
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end
% This function calculates the mass fraction of CsBr as a
% function of the Relative Humidity at a temperature of 25C
if RH > 1 
    error("RH should be 0 < RH < 1")
end 
if RH < 0.8475 || RH > 0.9482 
    error("below deliquescence relative humidity or above range") 
end  
A_4 = 10.31; 
A_3 = -17.57; 
A_2 = 10.36;
A_1 = -2.792; 
A_0 = 1.233;
f = @(xi) RH - A_0 - A_1.*xi - A_2.*xi.^2 - A_3.*xi.^3 - A_4.*xi.^4;
mf = robust_fzero(f, 0.2504, 0.5562, 0.4033);

end
% ---------------------------------------------------------
