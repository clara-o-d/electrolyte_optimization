function mf = calculate_mf_Na2SO4(RH)
% Add util folder to path if needed
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end
% This function calculates the mass fraction of Na2SO4 as a
% function of the Relative Humidity at a temperature of 25C
if RH > 1 
    error("RH should be 0 < RH < 1")
end 
if RH < 0.899 || RH > 0.9957 
    error("below deliquescence relative humidity or above range") 
end  
A_4 = -4.122; 
A_3 = 0.5557; 
A_2 = -0.01558;
A_1 = -0.2709; 
A_0 = 0.9994;
f = @(xi) RH - A_0 - A_1.*xi - A_2.*xi.^2 - A_3.*xi.^3 - A_4.*xi.^4;
mf = robust_fzero(f, 0.014, 0.2988, 0.1564);

end
% ---------------------------------------------------------
