function mf = calculate_mf_Xylitol(RH)
% Add util folder to path if needed
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end
% This function calculates the mass fraction of Xylitol (non-electrolyte) as a
% function of the Relative Humidity at a temperature of 25C
if RH > 1 
    error("RH should be 0 < RH < 1")
end 
if RH < 0.9201 || RH > 0.9988 
    error("below minimum relative humidity or above range") 
end  
A_4 = 0.031163; 
A_3 = -0.280282; 
A_2 = -0.089168;
A_1 = -0.119904; 
A_0 = 1.000011;
f = @(xi) RH - A_0 - A_1.*xi - A_2.*xi.^2 - A_3.*xi.^3 - A_4.*xi.^4;
mf = robust_fzero(f, 0.009987, 0.401691, 0.205839);

end
% ---------------------------------------------------------
