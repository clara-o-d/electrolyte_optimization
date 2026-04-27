function mf = calculate_mf_CuSO4(RH)
% Add util folder to path if needed
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end
% This function calculates the mass fraction of CuSO4 as a
% function of the Relative Humidity at a temperature of 25C
if RH > 1 
    error("RH should be 0 < RH < 1")
end 
if RH < 0.975 || RH > 0.9963 
    error("below deliquescence relative humidity or above range") 
end  
A_4 = -8.726; 
A_3 = 1.861; 
A_2 = -0.3016;
A_1 = -0.08611; 
A_0 = 0.9992;
f = @(xi) RH - A_0 - A_1.*xi - A_2.*xi.^2 - A_3.*xi.^3 - A_4.*xi.^4;
mf = robust_fzero(f, 0.0309, 0.1826, 0.1068);

end
% ---------------------------------------------------------
