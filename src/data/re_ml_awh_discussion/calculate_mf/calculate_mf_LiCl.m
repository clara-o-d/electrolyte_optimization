function mf = calculate_mf_LiCl(RH, T)
% Add util folder to path if needed
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end
% This function calculates the mass fraction of Lithium Chloride as a
% function of the vapor partial pressure and Temperature 
% Function from: https://doi.org/10.1016/j.ijthermalsci.2003.09.003

if RH > 1
    error('The relative humidity should be 0 < RH < 1');
end
if T > 100
    error('T too high. T should be given in C');
end

% Parameter for the vapour pressure equation   
p_0=0.28; p_1=4.3; p_2=0.60; p_3=0.21; p_4=5.10; p_5=0.49; p_6=0.362; 
p_7=-4.75; p_8=-0.40; p_9=0.03;   

theta = (T+273.15)/647; % Reduced temperature

f = @(xi) RH - (1 - (1 + (xi./p_6).^p_7).^p_8 - p_9.*exp(-(xi-0.1).^2./0.005)).* (2 - (1 + (xi./p_0).^p_1).^p_2 + ((1 + (xi./p_3).^p_4).^p_5 - 1).*theta);

mf = robust_fzero(f, 0.01, 0.75, 0.30);

end