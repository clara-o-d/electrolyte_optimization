% run_calculate_mf_batch.m
% Called from Python: matlab -batch "run_calculate_mf_batch('in.csv','out.csv')"
% in.csv: salt,RH,T  (header row)
% out.csv: salt,RH,T,mf  (mass fraction of salt)
%
% Uses load_salt_data and project calculate_mf functions.

function run_calculate_mf_batch(in_path, out_path)
    % Add paths
    [script_dir, ~, ~] = fileparts(mfilename('fullpath'));
    proj_root = fileparts(script_dir);
    addpath(fullfile(proj_root, 'calculate_mf'));
    addpath(fullfile(proj_root, 'util'));
    addpath(fullfile(proj_root, 'data'));

    % Load salt metadata
    salt_data = load_salt_data();

    % Build lookup: salt_name -> {func_name, func_args, MW, T_default}
    salt_lut = struct();
    for i = 1:length(salt_data)
        row = salt_data{i};
        name = row{1};
        salt_lut.(name) = struct('func', row{5}, 'func_args', row{6}, ...
            'MW', row{2}, 'T', row{9});
    end

    % Read input CSV (columns: salt, RH, T)
    tbl = readtable(in_path);
    col_names = tbl.Properties.VariableNames;
    salts = tbl.(col_names{1});
    RHs = tbl.(col_names{2});
    Ts = tbl.(col_names{3});

    mfs = zeros(size(RHs));
    for i = 1:length(RHs)
        s = salts(i);
        salt_name = char(string(s));
        RH = RHs(i);
        T = Ts(i);
        if ~isfield(salt_lut, salt_name)
            mfs(i) = NaN;
            continue;
        end
        info = salt_lut.(salt_name);
        try
            if info.func_args == 1
                mf = feval(info.func, RH, T);
            else
                mf = feval(info.func, RH);
            end
            mfs(i) = mf;
        catch
            mfs(i) = NaN;
        end
    end

    % Write output (column names must match Python reader)
    out_tbl = table(salts, RHs, Ts, mfs, 'VariableNames', {'salt','RH','T','mf'});
    writetable(out_tbl, out_path);
end
