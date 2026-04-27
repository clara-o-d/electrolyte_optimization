function mf = calculate_mf_KBr(RH)
% Add util folder to path if needed
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end
% This function calculates the mass fraction of KBr as a
% function of the Relative Humidity at a temperature of 25C
if RH > 1 
    error("RH should be 0 < RH < 1")
end 
if RH < 0.8325 || RH > 0.9528 
    error("below deliquescence relative humidity or above range") 
end  
A_4 = -265.3; 
A_3 = 234.2; 
A_2 = -76.14;
A_1 = 10.36; 
A_0 = 0.4537;
f = @(xi) RH - A_0 - A_1.*xi - A_2.*xi.^2 - A_3.*xi.^3 - A_4.*xi.^4;
mf = robust_fzero(f, 0.1299, 0.341, 0.2354);

end
% ---------------------------------------------------------
