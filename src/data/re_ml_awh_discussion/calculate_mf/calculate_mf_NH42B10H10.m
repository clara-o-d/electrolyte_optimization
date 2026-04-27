function mf = calculate_mf_NH42B10H10(RH)
% Add util folder to path if needed
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end
% This function calculates the mass fraction of NH4 as a
% function of the Relative Humidity at a temperature of 25C
if RH > 1 
    error("RH should be 0 < RH < 1")
end 
if RH < 0.8425 || RH > 0.9999 
    error("below deliquescence relative humidity or above range") 
end  
A_4 = 0.6987; 
A_3 = -0.7342; 
A_2 = -0.2297;
A_1 = -0.2755; 
A_0 = 0.9999;
f = @(xi) RH - A_0 - A_1.*xi - A_2.*xi.^2 - A_3.*xi.^3 - A_4.*xi.^4;
mf = robust_fzero(f, 0.0002, 0.3699, 0.1850);

end
% ---------------------------------------------------------
