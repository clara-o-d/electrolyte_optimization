function mf = calculate_mf_KCl(RH)
% Add util folder to path if needed
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end
% This function calculates the mass fraction of KCl as a
% function of the Relative Humidity at a temperature of 25C
if RH > 1 
    error("RH should be 0 < RH < 1")
end 
if RH < 0.852 || RH > 0.9935 
    error("below deliquescence relative humidity or above range") 
end  
A_4 = -0.1112; 
A_3 = -1.719; 
A_2 = -0.1585;
A_1 = -0.4398; 
A_0 = 1.0;
f = @(xi) RH - A_0 - A_1.*xi - A_2.*xi.^2 - A_3.*xi.^3 - A_4.*xi.^4;
mf = robust_fzero(f, 0.0147, 0.2512, 0.1329);

end
% ---------------------------------------------------------
