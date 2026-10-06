clear; close all;

%% Parameters
l = 1;          % domain side length [0,l]^3
epsilon = 1;    
mu = 1;         
t = 1;          
M_exact = 50;   
M_list = [2,4,6,8,10];  

% Yee-grid settings (matched to FDTD comparison)
N_grid = 128;    
h = l/N_grid;    

%% Build staggered Yee sampling coordinates for each field component
% Electric-field components (Ex/Ey/Ez) sampled at half-shifted locations
% Ex: (i+1/2, j, k)
[Ex_x, Ex_y, Ex_z] = meshgrid(h/2:h:l-h/2, 0:h:l, 0:h:l);
Ex_x = permute(Ex_x, [2,1,3]);
Ex_y = permute(Ex_y, [2,1,3]);
Ex_z = permute(Ex_z, [2,1,3]);

% Ey: (i, j+1/2, k)
[Ey_x, Ey_y, Ey_z] = meshgrid(0:h:l, h/2:h:l-h/2, 0:h:l);
Ey_x = permute(Ey_x, [2,1,3]);
Ey_y = permute(Ey_y, [2,1,3]);
Ey_z = permute(Ey_z, [2,1,3]);

% Ez: (i, j, k+1/2)
[Ez_x, Ez_y, Ez_z] = meshgrid(0:h:l, 0:h:l, h/2:h:l-h/2);
Ez_x = permute(Ez_x, [2,1,3]);
Ez_y = permute(Ez_y, [2,1,3]);
Ez_z = permute(Ez_z, [2,1,3]);

% Magnetic-field components (Hx/Hy/Hz) at complementary half-shifted locations
% Hx: (i, j+1/2, k+1/2)
[Hx_x, Hx_y, Hx_z] = meshgrid(0:h:l, h/2:h:l-h/2, h/2:h:l-h/2);
Hx_x = permute(Hx_x, [2,1,3]);
Hx_y = permute(Hx_y, [2,1,3]);
Hx_z = permute(Hx_z, [2,1,3]);

% Hy: (i+1/2, j, k+1/2)
[Hy_x, Hy_y, Hy_z] = meshgrid(h/2:h:l-h/2, 0:h:l, h/2:h:l-h/2);
Hy_x = permute(Hy_x, [2,1,3]);
Hy_y = permute(Hy_y, [2,1,3]);
Hy_z = permute(Hy_z, [2,1,3]);

% Hz: (i+1/2, j+1/2, k)
[Hz_x, Hz_y, Hz_z] = meshgrid(h/2:h:l-h/2, h/2:h:l-h/2, 0:h:l);
Hz_x = permute(Hz_x, [2,1,3]);
Hz_y = permute(Hz_y, [2,1,3]);
Hz_z = permute(Hz_z, [2,1,3]);

%% Storage arrays
err_E_Inf   = zeros(length(M_list), 1); 
order_E_Inf = zeros(length(M_list), 1); 
err_H_Inf   = zeros(length(M_list), 1); 
order_H_Inf = zeros(length(M_list), 1); 

%% Efficiency test (M = 10)
tic
HE_test_func = create_Maxwell_functions_simple(10, l, epsilon, mu);

fprintf('Numerical-solution function created, time: %.4fs\n', toc);
HE_test = HE_test_func(0.8,0.8,0.8,1);
fprintf('Sample-point data fetched, time: %.4fs\n', toc);

%% Build exact-solution function and evaluate it on the Yee grid
HE_exact_func = create_Maxwell_functions_simple(M_exact, l, epsilon, mu);

use_gpu = false;
gpu_dev = [];
if gpuDeviceCount() > 0
    use_gpu = true;
    gpu_dev = gpuDevice();
    fprintf('Using GPU acceleration, device: %s\n', gpu_dev.Name);

    
    Ex_x_gpu = gpuArray(Ex_x);  Ex_y_gpu = gpuArray(Ex_y);  Ex_z_gpu = gpuArray(Ex_z);
    Ey_x_gpu = gpuArray(Ey_x);  Ey_y_gpu = gpuArray(Ey_y);  Ey_z_gpu = gpuArray(Ey_z);
    Ez_x_gpu = gpuArray(Ez_x);  Ez_y_gpu = gpuArray(Ez_y);  Ez_z_gpu = gpuArray(Ez_z);
    Hx_x_gpu = gpuArray(Hx_x);  Hx_y_gpu = gpuArray(Hx_y);  Hx_z_gpu = gpuArray(Hx_z);
    Hy_x_gpu = gpuArray(Hy_x);  Hy_y_gpu = gpuArray(Hy_y);  Hy_z_gpu = gpuArray(Hy_z);
    Hz_x_gpu = gpuArray(Hz_x);  Hz_y_gpu = gpuArray(Hz_y);  Hz_z_gpu = gpuArray(Hz_z);
else
    warning('No GPU detected, falling back to CPU computation!');
    Ex_x_gpu = Ex_x;  Ex_y_gpu = Ex_y;  Ex_z_gpu = Ex_z;
    Ey_x_gpu = Ey_x;  Ey_y_gpu = Ey_y;  Ez_z_gpu = Ez_z;
    Ez_x_gpu = Ez_x;  Ez_y_gpu = Ez_y;  Ez_z_gpu = Ez_z;
    Hx_x_gpu = Hx_x;  Hx_y_gpu = Hx_y;  Hx_z_gpu = Hx_z;
    Hy_x_gpu = Hy_x;  Hy_y_gpu = Hy_y;  Hy_z_gpu = Hy_z;
    Hz_x_gpu = Hz_x;  Hz_y_gpu = Hz_y;  Hz_z_gpu = Hz_z;
end


fprintf('Pre-computing exact solution on the Yee grid (M_exact=%d, N_grid=%d)...\n', ...
    M_exact, N_grid);
tic;

% Exact solution: E components
HE_exact_Ex = HE_exact_func(Ex_x_gpu, Ex_y_gpu, Ex_z_gpu, t)';
E_exact_Ex = HE_exact_Ex{4};
HE_exact_Ey = HE_exact_func(Ey_x_gpu, Ey_y_gpu, Ey_z_gpu, t)';
E_exact_Ey = HE_exact_Ey{5};
HE_exact_Ez = HE_exact_func(Ez_x_gpu, Ez_y_gpu, Ez_z_gpu, t)';
E_exact_Ez = HE_exact_Ez{6};

% Exact solution: H components
HE_exact_Hx = HE_exact_func(Hx_x_gpu, Hx_y_gpu, Hx_z_gpu, t)';
H_exact_Hx = HE_exact_Hx{1};
HE_exact_Hy = HE_exact_func(Hy_x_gpu, Hy_y_gpu, Hy_z_gpu, t)';
H_exact_Hy = HE_exact_Hy{2};
HE_exact_Hz = HE_exact_func(Hz_x_gpu, Hz_y_gpu, Hz_z_gpu, t)';
H_exact_Hz = HE_exact_Hz{3};

if use_gpu
    E_exact_Ex = gather(E_exact_Ex);  E_exact_Ey = gather(E_exact_Ey);  E_exact_Ez = gather(E_exact_Ez);
    H_exact_Hx = gather(H_exact_Hx);  H_exact_Hy = gather(H_exact_Hy);  H_exact_Hz = gather(H_exact_Hz);
end

exact_time = toc;
fprintf('Exact solution computed, time: %.2fs\n', exact_time);

% Save exact-solution data to .mat file
save_filename = sprintf('ex2_exact_HEn_t=%.1f_M=%d_N=%d.mat', t, M_exact, N_grid);
save(save_filename, ...
     'E_exact_Ex', 'E_exact_Ey', 'E_exact_Ez', ...
     'H_exact_Hx', 'H_exact_Hy', 'H_exact_Hz', ...
     'M_exact', 'N_grid', 'l', 't', 'epsilon', 'mu');
fprintf('Exact solution saved to: %s\n', save_filename);

%% Loop over M to compute E/H inf-norm errors and convergence orders
fprintf('\nComputing E/H inf-norm errors and convergence orders for different M...\n');
fprintf('Yee grid: %d points per direction, spacing h=%.4f\n', N_grid, h);
fprintf('M list: %s\n\n', num2str(M_list));

total_numer_time = 0; 
for m_idx = 1:length(M_list)
    M = M_list(m_idx);
    fprintf('Computing M=%d...\n', M);
    tic;
    HE_func = create_Maxwell_functions_simple(M, l, epsilon, mu);

    HE_numer_Ex = HE_func(Ex_x_gpu, Ex_y_gpu, Ex_z_gpu, t)';
    E_numer_Ex = HE_numer_Ex{4};
    HE_numer_Ey = HE_func(Ey_x_gpu, Ey_y_gpu, Ey_z_gpu, t)';
    E_numer_Ey = HE_numer_Ey{5};
    HE_numer_Ez = HE_func(Ez_x_gpu, Ez_y_gpu, Ez_z_gpu, t)';
    E_numer_Ez = HE_numer_Ez{6};

    HE_numer_Hx = HE_func(Hx_x_gpu, Hx_y_gpu, Hx_z_gpu, t)';
    H_numer_Hx = HE_numer_Hx{1};
    HE_numer_Hy = HE_func(Hy_x_gpu, Hy_y_gpu, Hy_z_gpu, t)';
    H_numer_Hy = HE_numer_Hy{2};
    HE_numer_Hz = HE_func(Hz_x_gpu, Hz_y_gpu, Hz_z_gpu, t)';
    H_numer_Hz = HE_numer_Hz{3};

    if use_gpu
        E_numer_Ex = gather(E_numer_Ex);  E_numer_Ey = gather(E_numer_Ey);  E_numer_Ez = gather(E_numer_Ez);
        H_numer_Hx = gather(H_numer_Hx);  H_numer_Hy = gather(H_numer_Hy);  H_numer_Hz = gather(H_numer_Hz);
    end

    err_Ex = E_exact_Ex - E_numer_Ex;
    err_Ey = E_exact_Ey - E_numer_Ey;
    err_Ez = E_exact_Ez - E_numer_Ez;
    err_E_Inf(m_idx) = max([max(abs(err_Ex(:))), max(abs(err_Ey(:))), max(abs(err_Ez(:)))]);

    err_Hx = H_exact_Hx - H_numer_Hx;
    err_Hy = H_exact_Hy - H_numer_Hy;
    err_Hz = H_exact_Hz - H_numer_Hz;
    err_H_Inf(m_idx) = max([max(abs(err_Hx(:))), max(abs(err_Hy(:))), max(abs(err_Hz(:)))]);

    if m_idx >= 2
        M_last = M_list(m_idx-1);
        order_E_Inf(m_idx) = log(err_E_Inf(m_idx-1)/err_E_Inf(m_idx)) / log(M/M_last);
        order_H_Inf(m_idx) = log(err_H_Inf(m_idx-1)/err_H_Inf(m_idx)) / log(M/M_last);
    else
        order_E_Inf(m_idx) = NaN;
        order_H_Inf(m_idx) = NaN;
    end

    calc_time = toc;
    total_numer_time = total_numer_time + calc_time;
    fprintf('M=%d done, time: %.2fs | E_inf=%.3e | H_inf=%.3e\n', ...
        M, calc_time, err_E_Inf(m_idx), err_H_Inf(m_idx));
end

if use_gpu && ~isempty(gpu_dev)
    reset(gpu_dev);
    fprintf('\nGPU memory released; total numerical-solution time: %.2fs\n', total_numer_time);
end

%% Results table (E and H fields)
fprintf('\n=========================================================\n');
fprintf('     Fourier-series E/H inf-norm errors and orders on the Yee grid\n');
fprintf('===========================================================\n');
fprintf('Yee grid: %d points per direction, spacing h=%.4f\n', N_grid, h);
fprintf('Domain: [0,%.1f]^3, simulation time t=%.1fs\n\n', l, t);

result_data = [M_list', err_E_Inf, order_E_Inf, err_H_Inf, order_H_Inf];
result_cell = cell(length(M_list), 5);
for i = 1:length(M_list)
    result_cell{i,1} = num2str(result_data(i,1));          
    result_cell{i,2} = sprintf('%.3e', result_data(i,2));  
    if isnan(result_data(i,3))
        result_cell{i,3} = '--';
    else
        result_cell{i,3} = sprintf('%.3f', result_data(i,3)); 
    end
    result_cell{i,4} = sprintf('%.3e', result_data(i,4));  
    if isnan(result_data(i,5))
        result_cell{i,5} = '--';
    else
        result_cell{i,5} = sprintf('%.3f', result_data(i,5)); 
    end
end

T = cell2table(result_cell, ...
    'VariableNames', ...
    {'M', 'E_Inf_error', 'E_order_Inf', 'H_Inf_error', 'H_order_Inf'});
disp(T);

save_filename = sprintf('table_t=%.1f_N=%d_Order.mat', t, N_grid);
save(save_filename, 'T');
fprintf('Convergence data saved to file.\n');

%% Convergence visualization
figure('Color','white','Position',[100,100,700,400]);
loglog(M_list, err_E_Inf, 'ro-', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'E field');
hold on; grid on;
loglog(M_list, err_H_Inf, 'bs-', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'H field');
xlabel('Fourier truncation order M');
ylabel('Inf-norm error (log-log)');
title(sprintf('E/H inf-norm error decay (N_{grid}=%d, GPU-accelerated)', N_grid));
legend('Location','best');
set(gca, 'FontSize', 10);

figure('Color','white','Position',[100,100,700,400]);
plot(M_list(2:end), order_E_Inf(2:end), 'ro-', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'E');
hold on; grid on;
plot(M_list(2:end), order_H_Inf(2:end), 'bs-', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'H');
xlabel('M (N,K symmetric)');
ylabel('Order');
legend('Location','best');
set(gca, 'FontSize', 10);

fprintf('\n============================================================\n');
fprintf('Notes: 1. Convergence orders are based on E/H inf-norm errors (matched to FDTD comparison).\n');
fprintf('       2. Both exact and numerical solutions are GPU-accelerated; speed-up is significant at N_grid=128.\n');
fprintf('       3. The absolute slope of the error curve in log-log is approximately the convergence order.\n');
fprintf('       4. H-field convergence order is computed in the same way as E (log ratio of Fourier truncation orders).\n');
fprintf('       5. GPU memory is released automatically to avoid leaks.\n');
fprintf('============================================================\n');


function [E, H] = t_create_Maxwell_f_exact()
    E1 = @(x, y, z, t)(cos(2*sqrt(3)*pi*t)*cos(2*pi*x)*sin(2*pi*y)*sin(2*pi*z));
    E2 = @(x, y, z, t)-2*(cos(2*sqrt(3)*pi*t)*sin(2*pi*x)*cos(2*pi*y)*sin(2*pi*z));
    E3 = @(x, y, z, t)(cos(2*sqrt(3)*pi*t)*sin(2*pi*x)*sin(2*pi*y)*cos(2*pi*z));
    H1 = @(x, y, z, t)(-sqrt(3)*sin(2*sqrt(3)*pi*t)*sin(2*pi*x)*cos(2*pi*y)*cos(2*pi*z));
    H2 = @(x, y, z, t)0;
    H3 = @(x, y, z, t)(sqrt(3)*sin(2*sqrt(3)*pi*t)*cos(2*pi*x)*cos(2*pi*y)*sin(2*pi*z));
    E = {E1,E2,E3};
    H = {H1,H2,H3};
end
%% Build the 6x6 coefficient matrix A from the Fourier expansion
function A = create_A_matrix(m, n, k, mu, epsilon, l)
A = zeros(6, 6);
coeff = 2 * pi / l;

A(1, 5) =  1i * coeff * k / mu;
A(1, 6) = -1i * coeff * n / mu;
A(2, 4) = -1i * coeff * k / mu;
A(2, 6) =  1i * coeff * m / mu;
A(3, 4) =  1i * coeff * n / mu;
A(3, 5) = -1i * coeff * m / mu;
A(4, 2) = -1i * coeff * k / epsilon;
A(4, 3) =  1i * coeff * n / epsilon;
A(5, 1) =  1i * coeff * k / epsilon;
A(5, 3) = -1i * coeff * m / epsilon;
A(6, 1) = -1i * coeff * n / epsilon;
A(6, 2) =  1i * coeff * m / epsilon;

end

%% Pre-compute Fourier modes
function modes = precompute_modes_simple(M)
    m_list = -M:M;
    T_all = zeros(size(m_list));
    S_all = zeros(size(m_list));
    for idx = 1:length(m_list)
        m = m_list(idx);
        T_all(idx) = compute_T(m);
        S_all(idx) = compute_S(m);
    end

    modes = repmat(struct('m',0,'n',0,'k',0,'X0',zeros(6,1)), 1);

    mode_idx = 0;
    for m_idx = 1:length(m_list)
        m = m_list(m_idx);
        T_m = T_all(m_idx);
        S_m = S_all(m_idx);
        for n_idx = 1:length(m_list)
            n = m_list(n_idx);
            T_n = T_all(n_idx);
            S_n = S_all(n_idx);
            for k_idx = 1:length(m_list)
                k = m_list(k_idx);
                T_k = T_all(k_idx);
                S_k = S_all(k_idx);

                E1_coeff = -T_m * S_n * S_k;
                E2_coeff = 2 * S_m * T_n * S_k;
                E3_coeff = -S_m * S_n * T_k;

                tol = 1e-12;
                if abs(E1_coeff) < tol && abs(E2_coeff) < tol && abs(E3_coeff) < tol
                    continue;
                end
                mode_idx = mode_idx + 1;
                X0 = [0; 0; 0; E1_coeff; E2_coeff; E3_coeff];

                modes(mode_idx).m = m;
                modes(mode_idx).n = n;
                modes(mode_idx).k = k;
                modes(mode_idx).X0 = X0;
            end
        end
    end
end

%% Evaluate the field values from pre-computed modes
function HE_func = evaluate_field_simple(x, y, z, t, modes, l, mu, epsilon)
    if ~isequal(size(x), size(y), size(z))
        error('x, y, z must have identical sizes. x: %s, y: %s, z: %s', ...
            mat2str(size(x)), mat2str(size(y)), mat2str(size(z)));
    end

    sz = size(x);

    Ex = zeros(sz); Ey = zeros(sz); Ez = zeros(sz);
    Hx = zeros(sz); Hy = zeros(sz); Hz = zeros(sz);

    num_modes = length(modes);
    coeff_2pi_l = 2 * pi / l;
    for mode_idx = 1:num_modes
        m = modes(mode_idx).m;
        n = modes(mode_idx).n;
        k = modes(mode_idx).k;
        X0 = modes(mode_idx).X0;

        A = create_A_matrix(m, n, k, mu, epsilon, l);
        expA = expm(A *  t);
        Xt = expA * X0;
    
        phase = exp(1i * coeff_2pi_l * (m*x + n*y + k*z));

        Hx = Hx + real(Xt(1) * phase);
        Hy = Hy + real(Xt(2) * phase);
        Hz = Hz + real(Xt(3) * phase);
        Ex = Ex + real(Xt(4) * phase);
        Ey = Ey + real(Xt(5) * phase);
        Ez = Ez + real(Xt(6) * phase);
    end

    HE_func = {Hx, Hy, Hz, Ex, Ey, Ez};
end

%% Build a numerical E/H function handle for a given truncation M
function HE_func = create_Maxwell_functions_simple(M, l, epsilon, mu)
    modes = precompute_modes_simple(M);
    HE_func= @(x, y, z, t) evaluate_field_simple(x, y, z, t, modes, l, mu, epsilon);
end

%% Fourier coefficient T(m) of f(x) = exp(cos(2*pi*x)) on [0,1]
function T = compute_T(m)
    T = besseli(m, 1);
end
%% Fourier coefficient S(m) of g(x) = sin(2*pi*x)*exp(cos(2*pi*x)) on [0,1]
function S = compute_S(m)
    if m == 0
        S = 0;
    else
        S = -1i * m * besseli(m, 1);
    end
end
%% Algorithm-correctness helpers
% Fourier coefficients T(m) and S(m) of cos(2*pi*x) and sin(2*pi*x) on [0,1]
function T = compute_Ts(m)
    if abs(m) == 1
        T = 1/2;
    else
        T = 0;
    end
end
function S = compute_Ss(m)
    if m == -1
        S = 1i/2;
    elseif m == 1
        S = -1i/2;
    else
        S = 0;
    end
end
