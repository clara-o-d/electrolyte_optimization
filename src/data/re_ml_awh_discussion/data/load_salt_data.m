function salt_data = load_salt_data()
% LOAD_SALT_DATA Returns the canonical salt data dictionary.
%
% Canonical source for all salt metadata: molecular weight, valid RH ranges,
% calculate_mf function names, etc. Used by save_all_data_to_csv, plot scripts,
% and any other code needing salt humidity ranges.
%
% Returns: salt_data - cell array, each row:
%   {salt_name, MW, RH_min, RH_max, func_name, func_args, salt_type, valence,
%    temperature, nu, ionic_strength_factor, n_cat, n_an}
%
% func_args: 0 = no args, 1 = needs temperature T
% nu: ions per formula unit (e.g. NaCl->2, CaCl2->3)
% n_cat, n_an: stoichiometric subscripts for cations and anions (e.g. Na2SO4 -> 2,1)

salt_data = {
    % Endothermic salts (1:1 salts, I = m)
    {'NaCl', 58.443, 0.765, 0.99, 'calculate_mf_NaCl', 0, 1, 1, 25, 2, 1, 1, 1};
    {'KCl', 74.551, 0.855, 0.99, 'calculate_mf_KCl', 0, 1, 1, 25, 2, 1, 1, 1};
    {'NH4Cl', 53.491, 0.815, 0.99, 'calculate_mf_NH4Cl', 0, 1, 1, 25, 2, 1, 1, 1};
    {'CsCl', 168.363, 0.82, 0.99, 'calculate_mf_CsCl', 0, 1, 1, 25, 2, 1, 1, 1};
    {'NaNO3', 85.00, 0.971, 0.995, 'calculate_mf_NaNO3', 0, 1, 1, 25, 2, 1, 1, 1};
    {'AgNO3', 169.87, 0.865, 0.985, 'calculate_mf_AgNO3', 0, 1, 1, 25, 2, 1, 1, 1};
    {'KI', 165.998, 0.97, 0.995, 'calculate_mf_KI', 0, 1, 1, 25, 2, 1, 1, 1};
    {'LiNO3', 68.95, 0.736, 0.99, 'calculate_mf_LiNO3', 0, 1, 1, 25, 2, 1, 1, 1};
    {'KNO3', 101.10, 0.932, 0.995, 'calculate_mf_KNO3', 0, 1, 1, 25, 2, 1, 1, 1};
    {'NaClO4', 122.44, 0.778, 0.99, 'calculate_mf_NaClO4', 0, 1, 1, 25, 2, 1, 1, 1};
    {'KClO3', 122.55, 0.981, 0.9926, 'calculate_mf_KClO3', 0, 1, 1, 25, 2, 1, 1, 1};
    {'NaBr', 102.89, 0.614, 0.9280, 'calculate_mf_NaBr', 0, 1, 1, 30, 2, 1, 1, 1};
    {'NaI', 149.89, 0.581, 0.9659, 'calculate_mf_NaI', 0, 1, 1, 30, 2, 1, 1, 1};
    {'KBr', 119.00, 0.833, 0.9518, 'calculate_mf_KBr', 0, 1, 1, 30, 2, 1, 1, 1};
    {'RbCl', 120.92, 0.743, 0.9517, 'calculate_mf_RbCl', 0, 1, 1, 30, 2, 1, 1, 1};
    {'CsBr', 212.81, 0.848, 0.9472, 'calculate_mf_CsBr', 0, 1, 1, 30, 2, 1, 1, 1};
    {'CsI', 259.81, 0.913, 0.9614, 'calculate_mf_CsI', 0, 1, 1, 30, 2, 1, 1, 1};
    
    % Exothermic salts
    {'LiCl', 42.4, 0.12, 0.97, 'calculate_mf_LiCl', 1, 1, 1, 25, 2, 1, 1, 1};
    {'LiOH', 24, 0.85, 0.97, 'calculate_mf_LiOH', 0, 1, 1, 25, 2, 1, 1, 1};
    {'NaOH', 40, 0.23, 0.97, 'calculate_mf_NaOH', 0, 1, 1, 25, 2, 1, 1, 1};
    {'HCl', 36.5, 0.17, 0.97, 'calculate_mf_HCl', 0, 1, 1, 25, 2, 1, 1, 1};
    {'CaCl2', 111, 0.31, 0.97, 'calculate_mf_CaCl', 1, 1, 2, 25, 3, 3, 1, 2};
    {'MgCl2', 95.2, 0.33, 0.97, 'calculate_mf_MgCl', 0, 1, 2, 25, 3, 3, 1, 2};
    {'MgNO3', 148.3, 0.55, 0.9, 'calculate_mf_MgNO3', 0, 1, 2, 25, 3, 3, 1, 2};
    {'MgNO32', 148.3, 0.55, 0.9, 'calculate_mf_MgNO3', 0, 1, 2, 25, 3, 3, 1, 2};
    {'LiBr', 86.85, 0.07, 0.97, 'calculate_mf_LiBr', 0, 1, 1, 25, 2, 1, 1, 1};
    {'ZnCl2', 136.3, 0.07, 0.97, 'calculate_mf_ZnCl', 0, 1, 2, 25, 3, 3, 1, 2};
    {'ZnI2', 319.18, 0.25, 0.97, 'calculate_mf_ZnI', 0, 1, 2, 25, 3, 3, 1, 2};
    {'ZnBr2', 225.2, 0.08, 0.85, 'calculate_mf_ZnBr', 0, 1, 2, 25, 3, 3, 1, 2};
    {'LiI', 133.85, 0.18, 0.97, 'calculate_mf_LiI', 0, 1, 1, 25, 2, 1, 1, 1};
    
    % Sulfates
    {'Na2SO4', 142.04, 0.9000, 0.9947, 'calculate_mf_Na2SO4', 0, 2, 1, 25, 3, 3, 2, 1};
    {'K2SO4', 174.26, 0.9730, 0.9948, 'calculate_mf_K2SO4', 0, 2, 1, 25, 3, 3, 2, 1};
    {'NH42SO4', 132.14, 0.8320, 0.9949, 'calculate_mf_NH42SO4', 0, 2, 1, 25, 3, 3, 2, 1};
    {'MgSO4', 120.37, 0.9060, 0.9950, 'calculate_mf_MgSO4', 0, 1, 1, 25, 2, 4, 1, 1};
    {'MnSO4', 151.00, 0.9200, 0.9951, 'calculate_mf_MnSO4', 0, 1, 1, 25, 2, 4, 1, 1};
    {'Li2SO4', 109.94, 0.8540, 0.9946, 'calculate_mf_Li2SO4', 0, 2, 1, 25, 3, 3, 2, 1};
    {'NiSO4', 154.75, 0.9720, 0.9952, 'calculate_mf_NiSO4', 0, 1, 1, 25, 2, 4, 1, 1};
    {'CuSO4', 159.61, 0.9760, 0.9953, 'calculate_mf_CuSO4', 0, 1, 1, 25, 2, 4, 1, 1};
    {'ZnSO4', 161.44, 0.9390, 0.9952, 'calculate_mf_ZnSO4', 0, 1, 1, 25, 2, 4, 1, 1};
    
    % Nitrates (additional)
    {'NH4NO3', 80.043, 0.6266, 0.9967, 'calculate_mf_NH4NO3', 0, 1, 1, 25, 2, 1, 1, 1};
    {'BaNO3', 261.34, 0.9869, 0.9948, 'calculate_mf_BaNO32', 0, 1, 2, 25, 3, 3, 1, 2};
    {'CaNO3', 164.09, 0.6474, 0.9945, 'calculate_mf_CaNO32', 0, 1, 2, 25, 3, 3, 1, 2};
    
    % Halides (additional)
    {'CaBr2', 199.89, 0.6405, 0.9230, 'calculate_mf_CaBr2', 0, 1, 2, 30, 3, 3, 1, 2};
    {'CaI2', 293.89, 0.8331, 0.9514, 'calculate_mf_CaI2', 0, 1, 2, 30, 3, 3, 1, 2};
    {'SrCl2', 158.53, 0.8069, 0.9768, 'calculate_mf_SrCl2', 0, 1, 2, 30, 3, 3, 1, 2};
    % Updated Halides (User provided new data)
    {'SrI2', 341.43, 0.5589, 0.9361, 'calculate_mf_SrI2', 0, 1, 2, 25, 3, 3, 1, 2};
    {'BaCl2', 208.23, 0.9078, 0.9600, 'calculate_mf_BaCl2', 0, 1, 2, 25, 3, 3, 1, 2};
    {'BaBr2', 297.14, 0.7454, 0.9387, 'calculate_mf_BaBr2', 0, 1, 2, 25, 3, 3, 1, 2};
    {'SrBr2', 247.43, 0.7776, 0.9571, 'calculate_mf_SrBr2', 0, 1, 2, 25, 3, 3, 1, 2};

    % Chlorates and New Salts (Updated with polynomial fits - Feb 2026)
    {'LiClO4', 106.39, 0.7785, 0.9869, 'calculate_mf_LiClO4', 0, 1, 1, 25, 2, 1, 1, 1};
    {'NH4ClO4', 117.489, 0.9444, 0.9968, 'calculate_mf_NH4ClO4', 0, 1, 1, 25, 2, 1, 1, 1};
    {'NH42HPO4', 132.056, 0.9357, 0.9999, 'calculate_mf_NH42HPO4', 0, 2, 1, 25, 3, 3, 2, 1};
    {'CN3H62CO3', 180.166, 0.9439, 0.9999, 'calculate_mf_CN3H62CO3', 0, 2, 1, 25, 3, 3, 2, 1};
    {'NH42B10H10', 154.267, 0.8425, 0.9999, 'calculate_mf_NH42B10H10', 0, 2, 1, 25, 3, 3, 2, 1};
    {'Li2C6H4S2O6', 190.051, 0.8150, 0.9999, 'calculate_mf_Li2C6H4S2O6', 0, 2, 1, 25, 3, 3, 2, 1};
    {'C2H6S2O6', 190.198, 0.4996, 0.9999, 'calculate_mf_C2H6S2O6', 0, 1, 2, 25, 3, 3, 1, 2};  % Disulfonic acid (electrolyte)
    {'Na2S2O3', 158.11, 0.8072, 0.9999, 'calculate_mf_Na2S2O3', 0, 2, 1, 25, 3, 3, 2, 1};
    {'Na2C4H2O4', 160.036, 0.8696, 0.9999, 'calculate_mf_Na2C4H2O4', 0, 2, 1, 25, 3, 3, 2, 1};
    {'Na2C6H4S2O6', 282.204, 0.8280, 0.9999, 'calculate_mf_Na2C6H4S2O6', 0, 2, 1, 25, 3, 3, 2, 1};
    {'K2CrO4', 194.19, 0.8609, 0.9999, 'calculate_mf_K2CrO4', 0, 2, 1, 25, 3, 3, 2, 1};
    
    % Non-Electrolytes (Added Feb 2026 - organic compounds, sugars, alcohols)
    {'Glycerol', 92.09, 0.9120, 0.9940, 'calculate_mf_Glycerol', 0, 0, 0, 25, 0, 0, 1, 0};
    {'Sucrose', 342.3, 0.8355, 0.9946, 'calculate_mf_Sucrose', 0, 0, 0, 25, 0, 0, 1, 0};
    {'Glucose', 180.16, 0.9076, 0.9990, 'calculate_mf_Glucose', 0, 0, 0, 25, 0, 0, 1, 0};
    {'Urea', 60.06, 0.9070, 0.9950, 'calculate_mf_Urea', 0, 0, 0, 25, 0, 0, 1, 0};
    {'Fructose', 180.16, 0.3280, 0.9940, 'calculate_mf_Fructose', 0, 0, 0, 25, 0, 0, 1, 0};
    {'Ethanol', 46.07, 0.0630, 0.9640, 'calculate_mf_Ethanol', 0, 0, 0, 25, 0, 0, 1, 0};
    {'Methanol', 32.04, 0.0070, 0.9300, 'calculate_mf_Methanol', 0, 0, 0, 25, 0, 0, 1, 0};
    {'Xylose', 150.13, 0.9470, 0.9988, 'calculate_mf_Xylose', 0, 0, 0, 25, 0, 0, 1, 0};
    {'Arabinose', 150.13, 0.9419, 0.9994, 'calculate_mf_Arabinose', 0, 0, 0, 25, 0, 0, 1, 0};
    {'Xylitol', 152.15, 0.9201, 0.9988, 'calculate_mf_Xylitol', 0, 0, 0, 25, 0, 0, 1, 0};
    {'NH4Br', 97.942, 0.8080, 0.9967, 'calculate_mf_NH4Br', 0, 1, 1, 25, 2, 1, 1, 1};
    
    % Alkaline earth perchlorates (Paper 23: Osmotic and Activity Coefficients at 25°)
    {'CaClO42', 238.98, 0.2211, 0.9952, 'calculate_mf_CaClO42', 0, 1, 2, 25, 3, 3, 1, 2};
    {'SrClO42', 286.52, 0.2393, 0.9953, 'calculate_mf_SrClO42', 0, 1, 2, 25, 3, 3, 1, 2};
    {'BaClO42', 336.23, 0.5609, 0.9954, 'calculate_mf_BaClO42', 0, 1, 2, 25, 3, 3, 1, 2};
    
    % Chromium salts (sources_list_.xlsx - Chromium salt data at 25°)
    {'CrCl3', 158.36, 0.8548, 0.9941, 'calculate_mf_CrCl3', 0, 1, 3, 25, 4, 9, 1, 3};
    {'Cr2SO43', 392.16, 0.8531, 0.9963, 'calculate_mf_Cr2SO43', 0, 2, 3, 25, 5, 15, 2, 3};
    {'CrNO33', 238.011, 0.8481, 0.9943, 'calculate_mf_CrNO33', 0, 1, 3, 25, 4, 9, 1, 3};
    {'KCrSO42', 283.22, 0.9663, 0.9962, 'calculate_mf_KCrSO42', 0, 1, 3, 25, 4, 9, 1, 3};
    {'NH4CrSO42', 262.15, 0.9757, 0.9961, 'calculate_mf_NH4CrSO42', 0, 1, 3, 25, 4, 9, 1, 3};
    
    % Acids and strong electrolytes (Paper 25 in sources_list_.xlsx at 25°)
    {'NaF', 41.99, 0.9691, 0.9967, 'calculate_mf_NaF', 0, 1, 1, 25, 2, 1, 1, 1};
    {'HNO3', 63.01, 0.8827, 0.9966, 'calculate_mf_HNO3', 0, 1, 1, 25, 2, 1, 1, 1};
    {'HBr', 80.91, 0.9621, 0.9966, 'calculate_mf_HBr', 0, 1, 1, 25, 2, 1, 1, 1};
    {'HClO4', 100.46, 0.6343, 0.9966, 'calculate_mf_HClO4', 0, 1, 1, 25, 2, 1, 1, 1};
    {'NaClO3', 106.44, 0.8943, 0.9967, 'calculate_mf_NaClO3', 0, 1, 1, 25, 2, 1, 1, 1};
    {'HI', 127.91, 0.8471, 0.9966, 'calculate_mf_HI', 0, 1, 1, 25, 2, 1, 1, 1};
    {'NaBrO3', 150.89, 0.9311, 0.9967, 'calculate_mf_NaBrO3', 0, 1, 1, 25, 2, 1, 1, 1};
};
end