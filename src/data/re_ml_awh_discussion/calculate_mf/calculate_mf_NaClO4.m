function mf = calculate_mf_NaClO4(RH)
% Add util folder to path if needed
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end
% This function calculates the mass fraction of NaClO4 as a
% function of the Relative Humidity at a temperature of 25C
if RH > 1 
    error("RH should be 0 < RH < 1")
end 
if RH < 0.7775 || RH > 0.9934 
    error("below deliquescence relative humidity or above range") 
end  
A_4 = -1.533; 
A_3 = 0.4352; 
A_2 = -0.6407;
A_1 = -0.2401; 
A_0 = 0.996;
f = @(xi) RH - A_0 - A_1.*xi - A_2.*xi.^2 - A_3.*xi.^3 - A_4.*xi.^4;
mf = robust_fzero(f, 0.0208, 0.4088, 0.2148);

end
% ---------------------------------------------------------
