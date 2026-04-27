function mf = calculate_mf_NaBr(RH)
% Add util folder to path if needed
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end
% This function calculates the mass fraction of NaBr as a
% function of the Relative Humidity at a temperature of 25C
if RH > 1 
    error("RH should be 0 < RH < 1")
end 
if RH < 0.6133 || RH > 0.929 
    error("below deliquescence relative humidity or above range") 
end  
A_4 = -4.168; 
A_3 = -8.193; 
A_2 = 7.508;
A_1 = -2.657; 
A_0 = 1.208;
f = @(xi) RH - A_0 - A_1.*xi - A_2.*xi.^2 - A_3.*xi.^3 - A_4.*xi.^4;
mf = robust_fzero(f, 0.1709, 0.4509, 0.3109);

end
% ---------------------------------------------------------
