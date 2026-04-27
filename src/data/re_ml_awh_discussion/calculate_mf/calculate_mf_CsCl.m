function mf = calculate_mf_CsCl(RH)
% Add util folder to path if needed
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end
% This function calculates the mass fraction of CsCl as a
% function of the Relative Humidity at a temperature of 25C
if RH > 1 
    error("RH should be 0 < RH < 1")
end 
if RH < 0.817 || RH > 0.993 
    error("below deliquescence relative humidity or above range") 
end  
A_4 = -2.17; 
A_3 = 1.775; 
A_2 = -0.9289;
A_1 = -0.0579; 
A_0 = 0.9937;
f = @(xi) RH - A_0 - A_1.*xi - A_2.*xi.^2 - A_3.*xi.^3 - A_4.*xi.^4;
mf = robust_fzero(f, 0.0036, 0.5025, 0.253);

end
% ---------------------------------------------------------
