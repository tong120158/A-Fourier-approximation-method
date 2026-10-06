clear; close all; clc;

%% ==================== 例2：常系数 Maxwell 方程组 ====================
% 论文 RHF-Maxwell_new.tex 例2
%   Omega = (0,1)^3, epsilon = mu = 1
%   初值条件见论文 \eqref{ex2_t0}
%   使用论文引理1的解析解公式求解 ODE（不使用 expm）
%
% 截断参数 M = 2,4,6,8,10，参考精确解 M = 50
% 分别计算 t = 0.1s 和 t = 1.0s 两个时刻的误差，输出 LaTeX 表格
% 绘制 t=1.0s 时刻的 L^{\infty} 误差收敛对数图（说明超指数收敛）

%% ==================== 核心参数设置 ====================
l = 1;          % 区域边长
epsilon = 1;    % 介电常数
mu = 1;         % 磁导率
t_list = [0.1, 1.0];  % 两个计算时刻（分开计算，独立计时）
M_exact = 50;   % 参考精确解截断数
M_list = [2, 4, 6, 8, 10];  % 测试截断数
N_grid = 32;    % 误差计算网格分辨率（与论文一致 32^3）
h_grid = l / N_grid;

nM = length(M_list);
n_t = length(t_list);

%% ==================== 预计算精确解（M=M_exact，批量） ====================
fprintf('批量预计算精确解 (M=%d) 在 %d^3 网格上（%d 个时刻）...\n', M_exact, N_grid, n_t);
tic;
modes_exact = precompute_modes(M_exact);
E_exact_all = cell(1, n_t);
H_exact_all = cell(1, n_t);
for ti = 1:n_t
    [E_exact_all{ti}, H_exact_all{ti}] = evaluate_field_ifft(modes_exact, t_list(ti), N_grid, l, mu, epsilon);
end
fprintf('精确解完成，用时 %.2f 秒。\n\n', toc);

%% ==================== 分开计算各时刻（独立计时） ====================
results = struct();
for ti = 1:n_t
    t = t_list(ti);
    rkey = sprintf('t%d', round(t*10));
    results.(rkey).t = t;
    results.(rkey).M_list = M_list;
    results.(rkey).err_E.Linf = zeros(1, nM);
    results.(rkey).err_E.L2   = zeros(1, nM);
    results.(rkey).err_E.H1   = zeros(1, nM);
    results.(rkey).err_H.Linf = zeros(1, nM);
    results.(rkey).err_H.L2   = zeros(1, nM);
    results.(rkey).err_H.H1   = zeros(1, nM);
    results.(rkey).times      = zeros(1, nM);

    fprintf('\n========== t = %.1f s（独立计算）==========\n', t);
    for m_idx = 1:nM
        M = M_list(m_idx);
        fprintf('  计算 M=%d ...', M);
        tic;
        modes_M = precompute_modes(M);
        [E_grid, H_grid] = evaluate_field_ifft(modes_M, t, N_grid, l, mu, epsilon);
        elapsed = toc;
        results.(rkey).times(m_idx) = elapsed;

        [results.(rkey).err_E.Linf(m_idx), results.(rkey).err_E.L2(m_idx), results.(rkey).err_E.H1(m_idx)] = ...
            compute_error_norms(E_exact_all{ti}, E_grid, h_grid);
        [results.(rkey).err_H.Linf(m_idx), results.(rkey).err_H.L2(m_idx), results.(rkey).err_H.H1(m_idx)] = ...
            compute_error_norms(H_exact_all{ti}, H_grid, h_grid);

        fprintf(' %.2fs,  E_Linf=%.3e, H_Linf=%.3e\n', elapsed, ...
            results.(rkey).err_E.Linf(m_idx), results.(rkey).err_H.Linf(m_idx));
    end
end

%% ==================== 结果输出（LaTeX 表格格式） ====================
fprintf('\n============================================================\n');
fprintf('   误差与收敛阶（%d^3 网格，两个时刻）\n', N_grid);
fprintf('============================================================\n');

for ti = 1:n_t
    t = t_list(ti);
    r = results.(sprintf('t%d', round(t*10)));
    fprintf('\n----- t = %.1f s -----\n', t);

    % 电场 E
    fprintf('\n%% 电场 E (t=%.1fs)\n', t);
    for i = 1:nM
        oLinf = NaN; oL2 = NaN; oH1 = NaN;
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

    % 磁场 H
    fprintf('\n%% 磁场 H (t=%.1fs)\n', t);
    for i = 1:nM
        oLinf = NaN; oL2 = NaN; oH1 = NaN;
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
fprintf('\n============================================================\n');

%% ==================== 绘制 L^{\infty} 误差收敛对数图 ====================
% 纵轴对数刻度，展示误差随 M 的超指数衰减，无标题
% 增加代数2阶和指数参考线（见 result.md 二/三节），起点取 M=2 时的实测误差
% 取 t=1.0s 时刻的数据绘图（与论文 fig:ex2_convergence_log 对应）
r_t1 = results.t10;
err_E_Linf = r_t1.err_E.Linf;
err_H_Linf = r_t1.err_H.Linf;
fig = figure('Position', [100 100 700 500]);
hold on; grid on;
semilogy(M_list, max(err_E_Linf, 1e-16), '-o', 'Color', [0.000 0.447 0.741], ...
    'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', 'E');
semilogy(M_list, max(err_H_Linf, 1e-16), '-s', 'Color', [0.850 0.325 0.098], ...
    'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', 'H');

% 参考线：代数2阶 e(M) = e(M=2) * (2/M)^2
alg_ref_E = err_E_Linf(1) * (2 ./ M_list).^2;
alg_ref_H = err_H_Linf(1) * (2 ./ M_list).^2;
semilogy(M_list, alg_ref_E, '--', 'Color', [0.000 0.447 0.741], ...
    'LineWidth', 1.0, 'DisplayName', 'E:2阶代数收敛');
semilogy(M_list, alg_ref_H, '--', 'Color', [0.850 0.325 0.098], ...
    'LineWidth', 1.0, 'DisplayName', 'H:2阶代数收敛');

% 参考线：指数 e(M) = e(M=2) * exp(-2*(M-2))，即 result.md 中 c=2 的情形
exp_ref_E = err_E_Linf(1) * exp(-2*(M_list - 2));
exp_ref_H = err_H_Linf(1) * exp(-2*(M_list - 2));
semilogy(M_list, exp_ref_E, ':', 'Color', [0.000 0.447 0.741], ...
    'LineWidth', 1.0, 'DisplayName', 'E:2阶指数收敛');
semilogy(M_list, exp_ref_H, ':', 'Color', [0.850 0.325 0.098], ...
    'LineWidth', 1.0, 'DisplayName', 'H:2阶指数收敛');

xlabel('$M$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('$L^{\infty}$ error', 'Interpreter', 'latex', 'FontSize', 14);
set(gca, 'YScale', 'log', 'FontSize', 12);
legend('Interpreter', 'none', 'FontSize', 11, 'Location', 'best');
hold off;

% 保存到上级目录（work目录，与tex文件同级）
work_dir = '..';
saveas(fig, fullfile(work_dir, 'ex2_convergence_log.png'));
print(fig, fullfile(work_dir, 'ex2_convergence_log'), '-depsc');
fprintf('\n收敛对数图已保存：ex2_convergence_log.png / .eps\n');


%% ==================== 函数1：用论文引理1解析公式求解 ODE 并用 ifft 加速 ====================
function [E_grid, H_grid] = evaluate_field_ifft(modes, t, N, l, mu, epsilon)
    num_modes = length(modes);
    coeff_H = {zeros(N,N,N), zeros(N,N,N), zeros(N,N,N)};
    coeff_E = {zeros(N,N,N), zeros(N,N,N), zeros(N,N,N)};

    for mode_idx = 1:num_modes
        m1 = modes(mode_idx).m;
        m2 = modes(mode_idx).n;
        m3 = modes(mode_idx).k;
        X0 = modes(mode_idx).X0;

        Xt = solve_Xt_analytical(m1, m2, m3, t, X0, mu, epsilon, l);

        im = 1 + mod(m1, N);  in = 1 + mod(m2, N);  ik = 1 + mod(m3, N);
        coeff_H{1}(im,in,ik) = Xt(1);
        coeff_H{2}(im,in,ik) = Xt(2);
        coeff_H{3}(im,in,ik) = Xt(3);
        coeff_E{1}(im,in,ik) = Xt(4);
        coeff_E{2}(im,in,ik) = Xt(5);
        coeff_E{3}(im,in,ik) = Xt(6);
    end

    H_grid = cell(3,1);  E_grid = cell(3,1);
    for i = 1:3
        H_grid{i} = real(N^3 * ifftn(coeff_H{i}));
        E_grid{i} = real(N^3 * ifftn(coeff_E{i}));
    end
end

%% ==================== 函数2：论文引理1解析解 ====================
function Xt = solve_Xt_analytical(m1, m2, m3, t, X0, mu, epsilon, l)
    if m1 == 0 && m2 == 0 && m3 == 0
        Xt = X0;
        return;
    end

    abs_m = sqrt(m1^2 + m2^2 + m3^2);
    omega = 2i * pi * abs_m / (l * sqrt(epsilon * mu));

    if m1*m2 + m1*m3 + m2*m3 ~= 0
        alpha = sqrt(epsilon) / (sqrt(mu) * abs_m);
    else
        alpha = sqrt(epsilon) / sqrt(mu);
    end

    v1 = [m1; m2; m3; 0; 0; 0];
    v2 = [0; 0; 0; m1; m2; m3];
    [v3, v4, v5, v6] = compute_eigenvectors(m1, m2, m3, alpha);
    V = [v1, v2, v3, v4, v5, v6];

    c = V \ X0;

    emt = exp(-omega * t);
    ept = exp( omega * t);

    Xt = c(1)*v1 + c(2)*v2 ...
       + (c(3)*v3 + c(4)*v4) * emt ...
       + (c(5)*v5 + c(6)*v6) * ept;
end

%% ==================== 函数3：计算特征向量 v3~v6（论文表1） ====================
function [v3, v4, v5, v6] = compute_eigenvectors(m1, m2, m3, alpha)
    if m1 ~= 0 && m2 ~= 0 && m3 ~= 0
        v3 = [ alpha*m1*m2; -alpha*(m3^2+m1^2);  alpha*m3*m2; -m3; 0;  m1];
        v4 = [-alpha*m3*m1; -alpha*m3*m2;        alpha*(m1^2+m2^2); -m2; m1; 0];
        v5 = [-alpha*m1*m2;  alpha*(m3^2+m1^2); -alpha*m3*m2; -m3; 0;  m1];
        v6 = [ alpha*m3*m1;  alpha*m3*m2;       -alpha*(m1^2+m2^2); -m2; m1; 0];
    elseif m1 == 0 && m2 ~= 0 && m3 ~= 0
        v3 = [ alpha*(m3^2+m2^2); 0; 0; 0; -m3; m2];
        v4 = [0;  alpha*m3; -alpha*m2; 1; 0; 0];
        v5 = [-alpha*(m3^2+m2^2); 0; 0; 0; -m3; m2];
        v6 = [0; -alpha*m3;  alpha*m2; 1; 0; 0];
    elseif m1 ~= 0 && m2 == 0 && m3 ~= 0
        v3 = [0; -alpha*(m1^2+m3^2); 0; -m3; 0; m1];
        v4 = [-alpha*m3; 0;  alpha*m1; 0; 1; 0];
        v5 = [0;  alpha*(m1^2+m3^2); 0; -m3; 0; m1];
        v6 = [ alpha*m3; 0; -alpha*m1; 0; 1; 0];
    elseif m1 ~= 0 && m2 ~= 0 && m3 == 0
        v3 = [ alpha*m2; -alpha*m1; 0; 0; 0; 1];
        v4 = [0; 0;  alpha*(m1^2+m2^2); -m2; m1; 0];
        v5 = [-alpha*m2;  alpha*m1; 0; 0; 0; 1];
        v6 = [0; 0; -alpha*(m1^2+m2^2); -m2; m1; 0];
    elseif m1 == 0 && m2 == 0 && m3 ~= 0
        v3 = [-alpha; 0; 0; 0; 1; 0];
        v4 = [0;  alpha; 0; 1; 0; 0];
        v5 = [ alpha; 0; 0; 0; 1; 0];
        v6 = [0; -alpha; 0; 1; 0; 0];
    elseif m1 == 0 && m2 ~= 0 && m3 == 0
        v3 = [ alpha; 0; 0; 0; 0; 1];
        v4 = [0; 0; -alpha; 1; 0; 0];
        v5 = [-alpha; 0; 0; 0; 0; 1];
        v6 = [0; 0;  alpha; 1; 0; 0];
    else
        v3 = [0; -alpha; 0; 0; 0; 1];
        v4 = [0; 0;  alpha; 0; 1; 0];
        v5 = [0;  alpha; 0; 0; 0; 1];
        v6 = [0; 0; -alpha; 0; 1; 0];
    end
end

%% ==================== 函数4：等距网格上的离散三范数误差 ====================
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

%% ==================== 函数5：模式预计算（例2初值） ====================
function modes = precompute_modes(M)
    m_list = -M:M;
    T_all = zeros(size(m_list));
    S_all = zeros(size(m_list));
    for idx = 1:length(m_list)
        mm = m_list(idx);
        T_all(idx) = besseli(mm, 1);
        if mm == 0
            S_all(idx) = 0;
        else
            S_all(idx) = -1i * mm * besseli(mm, 1);
        end
    end

    modes = repmat(struct('m',0,'n',0,'k',0,'X0',zeros(6,1)), 1);
    mode_idx = 0;
    tol = 1e-14;
    for mi = 1:length(m_list)
        m = m_list(mi);  Tm = T_all(mi);  Sm = S_all(mi);
        for ni = 1:length(m_list)
            n = m_list(ni);  Tn = T_all(ni);  Sn = S_all(ni);
            for ki = 1:length(m_list)
                k = m_list(ki);  Tk = T_all(ki);  Sk = S_all(ki);

                E1 = -Tm * Sn * Sk;
                E2 =  2 * Sm * Tn * Sk;
                E3 = -Sm * Sn * Tk;

                if abs(E1) < tol && abs(E2) < tol && abs(E3) < tol
                    continue;
                end
                mode_idx = mode_idx + 1;
                modes(mode_idx).m = m;
                modes(mode_idx).n = n;
                modes(mode_idx).k = k;
                modes(mode_idx).X0 = [0; 0; 0; E1; E2; E3];
            end
        end
    end
end

%% ==================== 辅助函数 ====================
function s = fmt_order(o)
    if isnan(o)
        s = '--';
    else
        s = sprintf('%.3f', o);
    end
end
