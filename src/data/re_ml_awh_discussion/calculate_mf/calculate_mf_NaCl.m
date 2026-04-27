function mf = calculate_mf_NaCl(RH)
% Add util folder to path if needed
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end
% This function calculates the mass fraction of NaCl as a
% function of the Relative Humidity at a temperature of 25C
if RH > 1 
    error("RH should be 0 < RH < 1")
end 
if RH < 0.762 || RH > 0.9934 
    error("below deliquescence relative humidity or above range") 
end  
A_4 = 5.863; 
A_3 = -5.545; 
A_2 = -0.332;
A_1 = -0.5597; 
A_0 = 0.9998;
f = @(xi) RH - A_0 - A_1.*xi - A_2.*xi.^2 - A_3.*xi.^3 - A_4.*xi.^4;
mf = robust_fzero(f, 0.0116, 0.2596, 0.1356);

end
% ---------------------------------------------------------
