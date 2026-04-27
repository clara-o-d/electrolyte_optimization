function mf = calculate_mf_MgNO3(RH)
    if ~exist('robust_fzero', 'file')
        [filepath,~,~] = fileparts(mfilename('fullpath'));
        addpath(fullfile(filepath, '..', 'util'));
    end
    % This function calculates the mass fraction of Magnesium Nitrate as a
    % function of the Relative Humidity at a temperature of 20-24C
    % Fit on data from: https://doi.org/10.1080/027868299304219
    
    if RH > 1 
        error("RH should be 0 < RH < 1")
    end 
    A_4 = 0;
    A_3 = 6.075040; 
    A_2 = -8.649495; 
    A_1 = 1.944451;
    A_0 = 0.795876; 
    
    f = @(X) RH - A_0 - A_1.*X - A_2.*X.^2 - A_3.*X.^3 - A_4.*X.^4;
    mf = robust_fzero(f, 0.28, 0.80, 0.3);
    
    end