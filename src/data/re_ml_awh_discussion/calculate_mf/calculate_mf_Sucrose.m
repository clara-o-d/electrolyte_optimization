function mf = calculate_mf_Sucrose(RH)
% Add util folder to path if needed
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end
% This function calculates the mass fraction of Sucrose (non-electrolyte) as a
% function of the Relative Humidity at a temperature of 25C
if RH > 1 
    error("RH should be 0 < RH < 1")
end 
if RH < 0.8355 || RH > 0.9946 
    error("below minimum relative humidity or above range") 
end  
A_4 = -2.010672; 
A_3 = 2.135050; 
A_2 = -1.030039;
A_1 = 0.116700; 
A_0 = 0.990683;
f = @(xi) RH - A_0 - A_1.*xi - A_2.*xi.^2 - A_3.*xi.^3 - A_4.*xi.^4;
mf = robust_fzero(f, 0.087639, 0.678629, 0.383134);

end
% ---------------------------------------------------------
