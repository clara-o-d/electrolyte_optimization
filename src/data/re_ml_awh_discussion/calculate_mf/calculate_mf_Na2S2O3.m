function mf = calculate_mf_Na2S2O3(RH)
% Add util folder to path if needed
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end
% This function calculates the mass fraction of Na2S2O3 as a
% function of the Relative Humidity at a temperature of 25C
if RH > 1 
    error("RH should be 0 < RH < 1")
end 
if RH < 0.8072 || RH > 0.9999 
    error("below deliquescence relative humidity or above range") 
end  
A_4 = -4.9490; 
A_3 = 0.8416; 
A_2 = -0.1821;
A_1 = -0.2547; 
A_0 = 0.9998;
f = @(xi) RH - A_0 - A_1.*xi - A_2.*xi.^2 - A_3.*xi.^3 - A_4.*xi.^4;
mf = robust_fzero(f, 0.0002, 0.3905, 0.1953);

end
% ---------------------------------------------------------
