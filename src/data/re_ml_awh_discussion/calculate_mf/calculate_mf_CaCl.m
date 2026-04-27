function mf = calculate_mf_CaCl(RH, T)
% Add util folder to path if needed
if ~exist('robust_fzero', 'file')
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'util'));
end
% This function calculates the mass fraction of Calcium Chloride as a
% function of the vapor partial pressure and Temperature 
% Function from: https://doi.org/10.1016/j.ijthermalsci.2003.09.003

if RH > 1
    error('The relative humidity should be 0 < RH < 1');
end
if T > 100
    error('T too high. T should be given in C');
end

% Parameter for the vapour pressure equation   
q_0=0.31; q_1=3.698; q_2=0.60; q_3=0.231; q_4=4.584; q_5=0.49; q_6=0.478;
q_7=-5.20; q_8=-0.40; q_9=0.018;  

theta = (T+273.15)/647; % Reduced temperature

f = @(xi) RH - (1 - (1 + (xi./q_6).^q_7).^q_8 - q_9.*exp(-(xi-0.1).^2./0.005)).* (2 - (1 + (xi./q_0).^q_1).^q_2 + ((1 + (xi./q_3).^q_4).^q_5 - 1).*theta);

mf = robust_fzero(f, 0.01, 0.75, 0.30);

end
