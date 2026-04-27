function mf = calculate_mf_MnSO4(RH)
% Add util folder to path if needed
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end
% This function calculates the mass fraction of MnSO4 as a
% function of the Relative Humidity at a temperature of 25C
if RH > 1 
    error("RH should be 0 < RH < 1")
end 
if RH < 0.919 || RH > 0.9961 
    error("below deliquescence relative humidity or above range") 
end  
A_4 = -3.052; 
A_3 = -0.9499; 
A_2 = 0.1562;
A_1 = -0.1223; 
A_0 = 0.9996;
f = @(xi) RH - A_0 - A_1.*xi - A_2.*xi.^2 - A_3.*xi.^3 - A_4.*xi.^4;
mf = robust_fzero(f, 0.0293, 0.3118, 0.1706);

end
% ---------------------------------------------------------
