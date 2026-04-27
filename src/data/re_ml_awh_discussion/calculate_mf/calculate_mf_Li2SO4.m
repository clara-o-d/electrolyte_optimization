function mf = calculate_mf_Li2SO4(RH)
% Add util folder to path if needed
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end
% This function calculates the mass fraction of Li2SO4 as a
% function of the Relative Humidity at a temperature of 25C
if RH > 1 
    error("RH should be 0 < RH < 1")
end 
if RH < 0.853 || RH > 0.9956 
    error("below deliquescence relative humidity or above range") 
end  
A_4 = -5.224; 
A_3 = -0.6378; 
A_2 = -0.4414;
A_1 = -0.3625; 
A_0 = 0.9995;
f = @(xi) RH - A_0 - A_1.*xi - A_2.*xi.^2 - A_3.*xi.^3 - A_4.*xi.^4;
mf = robust_fzero(f, 0.0109, 0.248, 0.1295);

end
% ---------------------------------------------------------
