function x = robust_fzero(func, x_min, x_max, x_guess)
% Robust zero-finding using only base MATLAB (no toolboxes)
% Finds a valid bracket for fzero, or uses simple bisection

if nargin < 4
    x_guess = (x_min + x_max) / 2;
end

n_test = 20;
x_test = linspace(x_min, x_max, n_test);
f_test = arrayfun(func, x_test);

bracket_found = false;
for i = 1:(n_test-1)
    if sign(f_test(i)) ~= sign(f_test(i+1)) && ~isnan(f_test(i)) && ~isnan(f_test(i+1))
        try
            x = fzero(func, [x_test(i), x_test(i+1)]);
            bracket_found = true;
            break;
        catch
            continue;
        end
    end
end

if ~bracket_found
    try
        x = fzero(func, x_guess);
    catch
        [~, idx] = min(abs(f_test));
        x = x_test(idx);
    end
end

end

