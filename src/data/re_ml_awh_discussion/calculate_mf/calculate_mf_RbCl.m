function mf = calculate_mf_RbCl(RH)
% Add util folder to path if needed
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end
% This function calculates the mass fraction of RbCl as a
% function of the Relative Humidity at a temperature of 25C
if RH > 1 
    error("RH should be 0 < RH < 1")
end 
if RH < 0.7423 || RH > 0.9527 
    error("below deliquescence relative humidity or above range") 
end  
A_4 = -47.03; 
A_3 = 51.49; 
A_2 = -21.22;
A_1 = 3.355; 
A_0 = 0.7777;
f = @(xi) RH - A_0 - A_1.*xi - A_2.*xi.^2 - A_3.*xi.^3 - A_4.*xi.^4;
mf = robust_fzero(f, 0.1529, 0.4566, 0.3048);

end
% ---------------------------------------------------------
