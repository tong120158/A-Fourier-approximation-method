clear; close all; clc;
%% Parameters
l = 1;                      
t_list = [1.0, 50.0];        
M_exact = 50;               
M_list  = [2, 4, 8, 16];    
N_grid  = 16;               
N_slice = 64;               
N_fft   = 128;              
x3_slice = 0.5;             
h_grid  = l / N_grid;       

mu_func  = @(x1,x2,x3) 2 + sin(2*pi*x1).*sin(2*pi*x2).*sin(2*pi*x3);
eps_func = @(x1,x2,x3) 2 + cos(2*pi*x1).*cos(2*pi*x2).*cos(2*pi*x3);

fprintf('Computing Fourier coefficients of the initial condition (N_fft=%d, M=%d)...\n', N_fft, M_exact);
tic;
modes_all = compute_initial_modes_ex3(M_exact, N_fft, l);
fprintf('Done, %d nonzero modes, time: %.2fs.\n', length(modes_all), toc);

nM = length(M_list);
n_t = length(t_list);

fprintf('\nBatch reference solution (M=%d, %d time levels)...\n', M_exact, n_t);
tic;
modes_exact = filter_modes(modes_all, M_exact);
[E_exact_all, H_exact_all] = evaluate_field_variable_coeff_batch(modes_exact, t_list, N_grid, l, mu_func, eps_func);
fprintf('Reference solution done, time: %.2fs.\n', toc);

%% Per-time-level computation (timed independently)
results = struct();

for t_idx = 1:n_t
    t = t_list(t_idx);
    rkey = sprintf('t%d', round(t*10));
    results.(rkey) = struct( ...
        't', t, 'M_list', M_list, ...
        'err_E', struct('Linf', zeros(1,nM), 'L2', zeros(1,nM), 'H1', zeros(1,nM)), ...
        'err_H', struct('Linf', zeros(1,nM), 'L2', zeros(1,nM), 'H1', zeros(1,nM)), ...
        'times', zeros(1, nM));

    fprintf('\n========== t = %.1f s (timed independently) ==========\n', t);

    for m_idx = 1:nM
        M = M_list(m_idx);
        fprintf('  Computing M=%d ...', M);
        tic;
        modes_M = filter_modes(modes_all, M);
        [E_M, H_M] = evaluate_field_variable_coeff(modes_M, t, N_grid, l, mu_func, eps_func);
        elapsed = toc;
        results.(rkey).times(m_idx) = elapsed;

        % Compute errors
        [results.(rkey).err_E.Linf(m_idx), results.(rkey).err_E.L2(m_idx), results.(rkey).err_E.H1(m_idx)] = ...
            compute_error_norms(E_exact_all{t_idx}, E_M, h_grid);
        [results.(rkey).err_H.Linf(m_idx), results.(rkey).err_H.L2(m_idx), results.(rkey).err_H.H1(m_idx)] = ...
            compute_error_norms(H_exact_all{t_idx}, H_M, h_grid);

        fprintf(' %.2fs,  E_Linf=%.3e, H_Linf=%.3e\n', elapsed, ...
            results.(rkey).err_E.Linf(m_idx), results.(rkey).err_H.Linf(m_idx));
    end
end

%% High-resolution 2D slice (N_slice = 64)
fprintf('\n========== Computing 2D slice (N_slice=%d, x3=%.2f) ==========\n', N_slice, x3_slice);
t_slice = 50.0;
M_slice = 8;

% Numerical solution slice (M = 8)
fprintf('Computing M=%d numerical slice (N=%d, t=%.1fs)...\n', M_slice, N_slice, t_slice);
tic;
modes_M_slice = filter_modes(modes_all, M_slice);
[E_slice_M8, H_slice_M8] = evaluate_field_slice_2d(modes_M_slice, t_slice, N_slice, l, x3_slice, mu_func, eps_func);
fprintf('Done, time: %.2fs.\n', toc);

% Reference solution slice (M = 50)
fprintf('Computing M=%d reference slice (N=%d, t=%.1fs)...\n', M_exact, N_slice, t_slice);
tic;
[E_slice_exact, H_slice_exact] = evaluate_field_slice_2d(modes_exact, t_slice, N_slice, l, x3_slice, mu_func, eps_func);
fprintf('Done, time: %.2fs.\n', toc);

% Slice plot and error plot (using N_slice resolution)
plot_field_slice(E_slice_M8, H_slice_M8, N_slice, l, x3_slice);
plot_error_slice(E_slice_exact, H_slice_exact, E_slice_M8, H_slice_M8, N_slice, l, x3_slice);

%% L^infinity convergence plot at t = 50.0s
r_t1 = results.t500;
fig = figure('Position', [100 100 700 500]);
hold on; grid on;
semilogy(r_t1.M_list, max(r_t1.err_E.Linf, 1e-16), '-o', 'Color', [0.000 0.447 0.741], ...
    'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', 'E');
semilogy(r_t1.M_list, max(r_t1.err_H.Linf, 1e-16), '-s', 'Color', [0.850 0.325 0.098], ...
    'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', 'H');

alg_ref_E = r_t1.err_E.Linf(1) * (2 ./ r_t1.M_list).^2;
alg_ref_H = r_t1.err_H.Linf(1) * (2 ./ r_t1.M_list).^2;
semilogy(r_t1.M_list, alg_ref_E, '--', 'Color', [0.000 0.447 0.741], ...
    'LineWidth', 1.0, 'DisplayName', 'E: 2nd-order algebraic');
semilogy(r_t1.M_list, alg_ref_H, '--', 'Color', [0.850 0.325 0.098], ...
    'LineWidth', 1.0, 'DisplayName', 'H: 2nd-order algebraic');

exp_ref_E = r_t1.err_E.Linf(1) * exp(-(r_t1.M_list - 2));
exp_ref_H = r_t1.err_H.Linf(1) * exp(-(r_t1.M_list - 2));
semilogy(r_t1.M_list, exp_ref_E, ':', 'Color', [0.000 0.447 0.741], ...
    'LineWidth', 1.0, 'DisplayName', 'E: 1st-order exponential');
semilogy(r_t1.M_list, exp_ref_H, ':', 'Color', [0.850 0.325 0.098], ...
    'LineWidth', 1.0, 'DisplayName', 'H: 1st-order exponential');

xlabel('$M$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('$L^{\infty}$ error', 'Interpreter', 'latex', 'FontSize', 14);
set(gca, 'YScale', 'log', 'FontSize', 12);
legend('Interpreter', 'none', 'FontSize', 11, 'Location', 'best');
hold off;

work_dir = '..';
saveas(fig, fullfile(work_dir, 'ex3_convergence_log.png'));
print(fig, fullfile(work_dir, 'ex3_convergence_log'), '-depsc');
fprintf('Convergence plot saved: ex3_convergence_log.png / .eps\n');

print_results(results, t_list);

function modes = compute_initial_modes_ex3(M, N_fft, l)
    h = l / N_fft;
    x = (0:N_fft-1) * h;
    [X1, X2, X3] = ndgrid(x, x, x);

    eps_val = 2 + cos(2*pi*X1).*cos(2*pi*X2).*cos(2*pi*X3);
    exp_term = exp(sin(2*pi*X2) + sin(2*pi*X3));
    E01 = zeros(size(X1));
    E02 =  2*pi*cos(2*pi*X3).*exp_term ./ eps_val;
    E03 = -2*pi*cos(2*pi*X2).*exp_term ./ eps_val;

    clear X1 X2 X3 eps_val exp_term;
    coeff_E1 = fftn(E01) / N_fft^3;  clear E01;
    coeff_E2 = fftn(E02) / N_fft^3;  clear E02;
    coeff_E3 = fftn(E03) / N_fft^3;  clear E03;

    m_range = -M:M;
    indices = mod(m_range, N_fft) + 1;
    block_E1 = coeff_E1(indices, indices, indices);
    block_E2 = coeff_E2(indices, indices, indices);
    block_E3 = coeff_E3(indices, indices, indices);
    clear coeff_E1 coeff_E2 coeff_E3;

    tol = 1e-14;
    abs_sum = abs(block_E1) + abs(block_E2) + abs(block_E3);
    lin_idx = find(abs_sum > tol);
    [mi, mj, mk] = ind2sub(size(block_E1), lin_idx);

    n_modes = length(mi);
    e1_vals = block_E1(lin_idx);
    e2_vals = block_E2(lin_idx);
    e3_vals = block_E3(lin_idx);

    modes = repmat(struct('m',0,'n',0,'k',0,'X0',zeros(6,1)), 1, n_modes);
    for i = 1:n_modes
        modes(i).m = m_range(mi(i));
        modes(i).n = m_range(mj(i));
        modes(i).k = m_range(mk(i));
        modes(i).X0 = [0; 0; 0; e1_vals(i); e2_vals(i); e3_vals(i)];
    end
end


function modes_filtered = filter_modes(modes_all, M)
    keep = false(1, length(modes_all));
    for i = 1:length(modes_all)
        if abs(modes_all(i).m) <= M && abs(modes_all(i).n) <= M && abs(modes_all(i).k) <= M
            keep(i) = true;
        end
    end
    modes_filtered = modes_all(keep);
end

function [E_grid, H_grid] = evaluate_field_variable_coeff(modes, t, N, l, mu_func, eps_func)
    h = l / N;
    x = (0:N-1) * h;

    [X1, X2, X3] = ndgrid(x, x, x);
    mu_vec  = mu_func(X1(:), X2(:), X3(:));
    eps_vec = eps_func(X1(:), X2(:), X3(:));
    n_pts   = length(mu_vec);

    sqrt_mu  = sqrt(mu_vec);
    sqrt_eps = sqrt(eps_vec);
    sqrt_mue = sqrt_mu .* sqrt_eps;

    E_acc = zeros(3, n_pts);
    H_acc = zeros(3, n_pts);

    for mi = 1:length(modes)
        m1 = modes(mi).m;  m2 = modes(mi).n;  m3 = modes(mi).k;
        X0 = modes(mi).X0;
        E_init = X0(4:6);

        if m1 == 0 && m2 == 0 && m3 == 0
            Xt_all = repmat(X0, 1, n_pts);
        else
            abs_m = sqrt(m1^2 + m2^2 + m3^2);
            m_vec = [m1; m2; m3];
            omega_vec = 2i * pi * abs_m ./ (l * sqrt_mue);

            if m1*m2 + m1*m3 + m2*m3 ~= 0
                alpha_vec = sqrt_eps ./ (sqrt_mu * abs_m);
            else
                alpha_vec = sqrt_eps ./ sqrt_mu;
            end

            [a0, b, c0, d] = compute_abcd(m1, m2, m3);
            M2 = [m_vec, b, d];
            sol = M2 \ E_init;
            c2 = sol(1);  c3 = sol(2)/2;  c4 = sol(3)/2;

            S = c3 * a0 + c4 * c0;
            T = c3 * b  + c4 * d;
            c2m = c2 * m_vec;

            sinh_wt = sinh(omega_vec * t);
            cosh_wt = cosh(omega_vec * t);

            coeff_h = -2 * (alpha_vec .* sinh_wt);
            H_t = S .* coeff_h.';
            coeff_e = 2 * cosh_wt;
            E_t = c2m + T .* coeff_e.';
            Xt_all = [H_t; E_t];
        end

        phase1 = exp(2i*pi*m1*x/l);
        phase2 = exp(2i*pi*m2*x/l);
        phase3 = exp(2i*pi*m3*x/l);
        phase3d = reshape(phase1, N, 1, 1) .* reshape(phase2, 1, N, 1) .* reshape(phase3, 1, 1, N);
        phase_vec = phase3d(:);

        H_acc = H_acc + Xt_all(1:3, :) .* phase_vec.';
        E_acc = E_acc + Xt_all(4:6, :) .* phase_vec.';
    end

    H_grid = cell(3,1);  E_grid = cell(3,1);
    for i = 1:3
        H_grid{i} = reshape(real(H_acc(i, :)), N, N, N);
        E_grid{i} = reshape(real(E_acc(i, :)), N, N, N);
    end
end

function [E_results, H_results] = evaluate_field_variable_coeff_batch(modes, t_list, N, l, mu_func, eps_func)
    n_t = length(t_list);
    h = l / N;
    x = (0:N-1) * h;

    [X1, X2, X3] = ndgrid(x, x, x);
    mu_vec  = mu_func(X1(:), X2(:), X3(:));
    eps_vec = eps_func(X1(:), X2(:), X3(:));
    n_pts   = length(mu_vec);

    sqrt_mu  = sqrt(mu_vec);
    sqrt_eps = sqrt(eps_vec);
    sqrt_mue = sqrt_mu .* sqrt_eps;

    m_max = max(abs([[modes.m], [modes.n], [modes.k]]));
    if isempty(m_max), m_max = 0; end
    phase_table = cell(1, 2*m_max+1);
    for idx = 1:(2*m_max+1)
        mm = idx - m_max - 1;
        phase_table{idx} = exp(2i*pi*mm*x/l);
    end

    E_acc = cell(1, n_t);
    H_acc = cell(1, n_t);
    for ti = 1:n_t
        E_acc{ti} = zeros(3, n_pts);
        H_acc{ti} = zeros(3, n_pts);
    end

    for mi = 1:length(modes)
        m1 = modes(mi).m;  m2 = modes(mi).n;  m3 = modes(mi).k;
        X0 = modes(mi).X0;
        E_init = X0(4:6);

        phase1 = phase_table{m1 + m_max + 1};
        phase2 = phase_table{m2 + m_max + 1};
        phase3 = phase_table{m3 + m_max + 1};
        phase3d = reshape(phase1, N, 1, 1) .* reshape(phase2, 1, N, 1) .* reshape(phase3, 1, 1, N);
        phase_vec = phase3d(:);

        if m1 == 0 && m2 == 0 && m3 == 0
            Xt_all = repmat(X0, 1, n_pts);
            for ti = 1:n_t
                H_acc{ti} = H_acc{ti} + Xt_all(1:3, :) .* phase_vec.';
                E_acc{ti} = E_acc{ti} + Xt_all(4:6, :) .* phase_vec.';
            end
        else
            abs_m = sqrt(m1^2 + m2^2 + m3^2);
            m_vec = [m1; m2; m3];
            omega_vec = 2i * pi * abs_m ./ (l * sqrt_mue);

            if m1*m2 + m1*m3 + m2*m3 ~= 0
                alpha_vec = sqrt_eps ./ (sqrt_mu * abs_m);
            else
                alpha_vec = sqrt_eps ./ sqrt_mu;
            end

            [a0, b, c0, d] = compute_abcd(m1, m2, m3);
            M2 = [m_vec, b, d];
            sol = M2 \ E_init;
            c2 = sol(1);  c3 = sol(2)/2;  c4 = sol(3)/2;

            S = c3 * a0 + c4 * c0;
            T = c3 * b  + c4 * d;
            c2m = c2 * m_vec;

            for ti = 1:n_t
                t = t_list(ti);
                sinh_wt = sinh(omega_vec * t);
                cosh_wt = cosh(omega_vec * t);

                coeff_h = -2 * (alpha_vec .* sinh_wt);
                H_t = S .* coeff_h.';
                coeff_e = 2 * cosh_wt;
                E_t = c2m + T .* coeff_e.';

                H_acc{ti} = H_acc{ti} + H_t .* phase_vec.';
                E_acc{ti} = E_acc{ti} + E_t .* phase_vec.';
            end
        end
    end

    E_results = cell(n_t, 1);
    H_results = cell(n_t, 1);
    for ti = 1:n_t
        E_results{ti} = cell(3,1);
        H_results{ti} = cell(3,1);
        for i = 1:3
            H_results{ti}{i} = reshape(real(H_acc{ti}(i, :)), N, N, N);
            E_results{ti}{i} = reshape(real(E_acc{ti}(i, :)), N, N, N);
        end
    end
end

function [E_slice, H_slice] = evaluate_field_slice_2d(modes, t, N, l, x3_val, mu_func, eps_func)
    h = l / N;
    x = (0:N-1) * h;
    [X1, X2] = ndgrid(x, x);

    mu_vec  = mu_func(X1(:), X2(:), x3_val);
    eps_vec = eps_func(X1(:), X2(:), x3_val);
    n_pts   = length(mu_vec);

    sqrt_mu  = sqrt(mu_vec);
    sqrt_eps = sqrt(eps_vec);
    sqrt_mue = sqrt_mu .* sqrt_eps;

    E_acc = zeros(3, n_pts);
    H_acc = zeros(3, n_pts);

    for mi = 1:length(modes)
        m1 = modes(mi).m;  m2 = modes(mi).n;  m3 = modes(mi).k;
        X0 = modes(mi).X0;
        E_init = X0(4:6);

        if m1 == 0 && m2 == 0 && m3 == 0
            Xt_all = repmat(X0, 1, n_pts);
        else
            abs_m = sqrt(m1^2 + m2^2 + m3^2);
            m_vec = [m1; m2; m3];
            omega_vec = 2i * pi * abs_m ./ (l * sqrt_mue);

            if m1*m2 + m1*m3 + m2*m3 ~= 0
                alpha_vec = sqrt_eps ./ (sqrt_mu * abs_m);
            else
                alpha_vec = sqrt_eps ./ sqrt_mu;
            end

            [a0, b, c0, d] = compute_abcd(m1, m2, m3);
            M2 = [m_vec, b, d];
            sol = M2 \ E_init;
            c2 = sol(1);  c3 = sol(2)/2;  c4 = sol(3)/2;

            S = c3 * a0 + c4 * c0;
            T = c3 * b  + c4 * d;
            c2m = c2 * m_vec;

            sinh_wt = sinh(omega_vec * t);
            cosh_wt = cosh(omega_vec * t);

            coeff_h = -2 * (alpha_vec .* sinh_wt);
            H_t = S .* coeff_h.';
            coeff_e = 2 * cosh_wt;
            E_t = c2m + T .* coeff_e.';
            Xt_all = [H_t; E_t];
        end

        phase1 = exp(2i*pi*m1*x/l);
        phase2 = exp(2i*pi*m2*x/l);
        phase3_scalar = exp(2i*pi*m3*x3_val/l);
        phase2d = reshape(phase1, N, 1) .* reshape(phase2, 1, N) .* phase3_scalar;
        phase_vec = phase2d(:);

        H_acc = H_acc + Xt_all(1:3, :) .* phase_vec.';
        E_acc = E_acc + Xt_all(4:6, :) .* phase_vec.';
    end

    H_slice = cell(3,1);  E_slice = cell(3,1);
    for i = 1:3
        H_slice{i} = reshape(real(H_acc(i, :)), N, N);
        E_slice{i} = reshape(real(E_acc(i, :)), N, N);
    end
end


function [a0, b, c0, d] = compute_abcd(m1, m2, m3)
    if m1 ~= 0 && m2 ~= 0 && m3 ~= 0
        a0 = [m1*m2; -(m3^2+m1^2); m3*m2];
        b  = [-m3; 0; m1];
        c0 = [-m3*m1; -m3*m2; m1^2+m2^2];
        d  = [-m2; m1; 0];
    elseif m1 == 0 && m2 ~= 0 && m3 ~= 0
        a0 = [m3^2+m2^2; 0; 0];
        b  = [0; -m3; m2];
        c0 = [0; m3; -m2];
        d  = [1; 0; 0];
    elseif m1 ~= 0 && m2 == 0 && m3 ~= 0
        a0 = [0; -(m1^2+m3^2); 0];
        b  = [-m3; 0; m1];
        c0 = [-m3; 0; m1];
        d  = [0; 1; 0];
    elseif m1 ~= 0 && m2 ~= 0 && m3 == 0
        a0 = [m2; -m1; 0];
        b  = [0; 0; 1];
        c0 = [0; 0; m1^2+m2^2];
        d  = [-m2; m1; 0];
    elseif m1 == 0 && m2 == 0 && m3 ~= 0
        a0 = [-1; 0; 0];
        b  = [0; 1; 0];
        c0 = [0; 1; 0];
        d  = [1; 0; 0];
    elseif m1 == 0 && m2 ~= 0 && m3 == 0
        a0 = [1; 0; 0];
        b  = [0; 0; 1];
        c0 = [0; 0; -1];
        d  = [1; 0; 0];
    else
        a0 = [0; -1; 0];
        b  = [0; 0; 1];
        c0 = [0; 0; 1];
        d  = [0; 1; 0];
    end
end


function [err_Linf, err_L2, err_H1] = compute_error_norms(U_exact, U_h, h)
    dV = h^3;
    diff_L2_sq = 0;  diff_gr_sq = 0;  Linf_max = 0;
    for i = 1:3
        e = U_exact{i} - U_h{i};
        diff_L2_sq = diff_L2_sq + sum(e(:).^2);
        Linf_max = max(Linf_max, max(abs(e(:))));
        [ex, ey, ez] = gradient(e, h, h, h);
        diff_gr_sq = diff_gr_sq + sum(ex(:).^2) + sum(ey(:).^2) + sum(ez(:).^2);
    end
    err_Linf = Linf_max;
    err_L2   = sqrt(dV * diff_L2_sq);
    err_H1   = sqrt(dV * (diff_L2_sq + diff_gr_sq));
end

function plot_field_slice(E_slice, H_slice, N, l, x3_val)
    h = l / N;
    x = (0:N-1) * h;

    fig = figure('Position', [50 50 1400 800]);
    comp_names = {'$E_1$', '$E_2$', '$E_3$', '$H_1$', '$H_2$', '$H_3$'};
    fields = {E_slice, H_slice};

    for fi = 1:2
        for ci = 1:3
            subplot(2, 3, (fi-1)*3 + ci);
            imagesc(x, x, fields{fi}{ci});
            colorbar;
            xlabel('$x_1$', 'Interpreter', 'latex', 'FontSize', 12);
            ylabel('$x_2$', 'Interpreter', 'latex', 'FontSize', 12);
            title(comp_names{(fi-1)*3+ci}, 'Interpreter', 'latex', 'FontSize', 14);
            axis equal tight;
        end
    end

    work_dir = '..';
    saveas(fig, fullfile(work_dir, 'ex3_field_slice.png'));
    print(fig, fullfile(work_dir, 'ex3_field_slice'), '-depsc');
    fprintf('Slice figure saved: ex3_field_slice.png / .eps\n');
end

function plot_error_slice(E_exact, H_exact, E_M, H_M, N, l, x3_val)
    h = l / N;
    x = (0:N-1) * h;

    err_E = cell(3,1);  err_H = cell(3,1);
    for i = 1:3
        err_E{i} = abs(E_exact{i} - E_M{i});
        err_H{i} = abs(H_exact{i} - H_M{i});
    end

    fig = figure('Position', [50 50 1400 800]);
    comp_names = {'$|E_1 - E_1^M|$', '$|E_2 - E_2^M|$', '$|E_3 - E_3^M|$', ...
                  '$|H_1 - H_1^M|$', '$|H_2 - H_2^M|$', '$|H_3 - H_3^M|$'};
    errs = {err_E, err_H};

    for fi = 1:2
        for ci = 1:3
            subplot(2, 3, (fi-1)*3 + ci);
            imagesc(x, x, errs{fi}{ci});
            colorbar;
            xlabel('$x_1$', 'Interpreter', 'latex', 'FontSize', 12);
            ylabel('$x_2$', 'Interpreter', 'latex', 'FontSize', 12);
            title(comp_names{(fi-1)*3+ci}, 'Interpreter', 'latex', 'FontSize', 14);
            axis equal tight;
        end
    end

    work_dir = '..';
    saveas(fig, fullfile(work_dir, 'ex3_error_slice.png'));
    print(fig, fullfile(work_dir, 'ex3_error_slice'), '-depsc');
    fprintf('Error-slice figure saved: ex3_error_slice.png / .eps\n');
end

function print_results(results, t_list)
    fprintf('\n\n==================== LaTeX table data ====================\n');
    for t_idx = 1:length(t_list)
        t = t_list(t_idx);
        rkey_print = sprintf('t%d', round(t*10));
        r = results.(rkey_print);
        M_list = r.M_list;
        nM = length(M_list);

        fprintf('\n----- t = %.1f s -----\n', t);

        fprintf('\n%% Electric field E (t=%.1fs)\n', t);
        for i = 1:nM
            oLinf = NaN;  oL2 = NaN;  oH1 = NaN;
            if i > 1
                oLinf = log(r.err_E.Linf(i-1)/r.err_E.Linf(i)) / log(M_list(i)/M_list(i-1));
                oL2   = log(r.err_E.L2(i-1)/r.err_E.L2(i))   / log(M_list(i)/M_list(i-1));
                oH1   = log(r.err_E.H1(i-1)/r.err_E.H1(i))   / log(M_list(i)/M_list(i-1));
            end
            fprintf('M=%d & %.3e & %s & %.3e & %s & %.3e & %s & %.2f \\\\\n', ...
                M_list(i), r.err_E.Linf(i), fmt_order(oLinf), ...
                r.err_E.L2(i), fmt_order(oL2), ...
                r.err_E.H1(i), fmt_order(oH1), r.times(i));
        end

        fprintf('\n%% Magnetic field H (t=%.1fs)\n', t);
        for i = 1:nM
            oLinf = NaN;  oL2 = NaN;  oH1 = NaN;
            if i > 1
                oLinf = log(r.err_H.Linf(i-1)/r.err_H.Linf(i)) / log(M_list(i)/M_list(i-1));
                oL2   = log(r.err_H.L2(i-1)/r.err_H.L2(i))   / log(M_list(i)/M_list(i-1));
                oH1   = log(r.err_H.H1(i-1)/r.err_H.H1(i))   / log(M_list(i)/M_list(i-1));
            end
            fprintf('M=%d & %.3e & %s & %.3e & %s & %.3e & %s & %.2f \\\\\n', ...
                M_list(i), r.err_H.Linf(i), fmt_order(oLinf), ...
                r.err_H.L2(i), fmt_order(oL2), ...
                r.err_H.H1(i), fmt_order(oH1), r.times(i));
        end
    end
    fprintf('\n==================== End of table data ====================\n');
end

function s = fmt_order(o)
    if isnan(o)
        s = '--';
    else
        s = sprintf('%.3f', o);
    end
end
