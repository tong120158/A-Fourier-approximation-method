clear; close all;

%% ======================= 核心参数设置 ===========================
l = 1;          % 区域边长 [0,l]^3
epsilon = 1;    % 介电常数
mu = 1;         % 磁导率
t = 1;          % 计算时间
M_exact = 50;   % 精确解的截断数
M_list = [2,4,6,8,10];  % 需测试的Fourier截断数

% ===== Yee网格参数（适配FDTD对比）=====
N_grid = 128;    % 单方向Yee网格数
h = l/N_grid;   % 网格步长

%% =================== 生成Yee网格各分量的采样坐标 =================
% -------- 1. 电场分量坐标（Ex/Ey/Ez）--------
% Ex: (i+0.5,j,k) → x=h/2:h:l-h/2, y=0:h:l, z=0:h:l
[Ex_x, Ex_y, Ex_z] = meshgrid(h/2:h:l-h/2, 0:h:l, 0:h:l);
% 置换维度：交换第一/第二维，保留物理意义，
Ex_x = permute(Ex_x, [2,1,3]);
Ex_y = permute(Ex_y, [2,1,3]);
Ex_z = permute(Ex_z, [2,1,3]);

% Ey: (i,j+0.5,k) → x=0:h:l, y=h/2:h:l-h/2, z=0:h:l
[Ey_x, Ey_y, Ey_z] = meshgrid(0:h:l, h/2:h:l-h/2, 0:h:l);
Ey_x = permute(Ey_x, [2,1,3]);
Ey_y = permute(Ey_y, [2,1,3]);
Ey_z = permute(Ey_z, [2,1,3]);

% Ez: (i,j,k+0.5) → x=0:h:l, y=0:h:l, z=h/2:h:l-h/2
[Ez_x, Ez_y, Ez_z] = meshgrid(0:h:l, 0:h:l, h/2:h:l-h/2);
Ez_x = permute(Ez_x, [2,1,3]);
Ez_y = permute(Ez_y, [2,1,3]);
Ez_z = permute(Ez_z, [2,1,3]);

% -------- 2. 磁场分量坐标（Hx/Hy/Hz）--------
% Hx: (i,j+0.5,k+0.5) → x=0:h:l, y=h/2:h:l-h/2, z=h/2:h:l-h/2
[Hx_x, Hx_y, Hx_z] = meshgrid(0:h:l, h/2:h:l-h/2, h/2:h:l-h/2);
Hx_x = permute(Hx_x, [2,1,3]);
Hx_y = permute(Hx_y, [2,1,3]);
Hx_z = permute(Hx_z, [2,1,3]);

% Hy: (i+0.5,j,k+0.5) → x=h/2:h:l-h/2, y=0:h:l, z=h/2:h:l-h/2
[Hy_x, Hy_y, Hy_z] = meshgrid(h/2:h:l-h/2, 0:h:l, h/2:h:l-h/2);
Hy_x = permute(Hy_x, [2,1,3]);
Hy_y = permute(Hy_y, [2,1,3]);
Hy_z = permute(Hy_z, [2,1,3]);

% Hz: (i+0.5,j+0.5,k) → x=h/2:h:l-h/2, y=h/2:h:l-h/2, z=0:h:l
[Hz_x, Hz_y, Hz_z] = meshgrid(h/2:h:l-h/2, h/2:h:l-h/2, 0:h:l);
Hz_x = permute(Hz_x, [2,1,3]);
Hz_y = permute(Hz_y, [2,1,3]);
Hz_z = permute(Hz_z, [2,1,3]);

%% ======================= 初始化存储数组 =========================
err_E_Inf = zeros(length(M_list), 1);  % E场无穷范数误差
order_E_Inf = zeros(length(M_list), 1);% E场收敛阶（基于Inf误差）
err_H_Inf = zeros(length(M_list), 1);  % H场无穷范数误差 % 新增
order_H_Inf = zeros(length(M_list), 1);% H场收敛阶（基于Inf误差）% 新增

%% ================= 测试算法的效率（M=10） =======================
tic
HE_test_func = create_Maxwell_functions_simple(10, l, epsilon, mu);

fprintf('数值解函数创建完成，用时：%.4f秒\n', toc);
HE_test = HE_test_func(0.8,0.8,0.8,1);
fprintf('采样点数据获取完成，用时：%.4f秒\n', toc);

%% ============== 加载精确解函数并计算Yee网格精确解=================
HE_exact_func = create_Maxwell_functions_simple(M_exact, l, epsilon, mu);

% ===== 预检查GPU是否可用（修复if-else语法结构）=====
use_gpu = false;  % 先初始化默认值
gpu_dev = [];     % 初始化GPU设备变量
if gpuDeviceCount() > 0  % 修正判断条件，更直观
    use_gpu = true;
    gpu_dev = gpuDevice(); % 激活默认GPU
    fprintf('使用GPU加速计算，设备名称：%s\n', gpu_dev.Name);
    
    % -------- 提前创建所有坐标的GPU数组（供精确解+数值解复用）--------
    Ex_x_gpu = gpuArray(Ex_x);
    Ex_y_gpu = gpuArray(Ex_y);
    Ex_z_gpu = gpuArray(Ex_z);
    
    Ey_x_gpu = gpuArray(Ey_x);
    Ey_y_gpu = gpuArray(Ey_y);
    Ey_z_gpu = gpuArray(Ey_z);
    
    Ez_x_gpu = gpuArray(Ez_x);
    Ez_y_gpu = gpuArray(Ez_y);
    Ez_z_gpu = gpuArray(Ez_z);
    
    Hx_x_gpu = gpuArray(Hx_x);
    Hx_y_gpu = gpuArray(Hx_y);
    Hx_z_gpu = gpuArray(Hx_z);
    
    Hy_x_gpu = gpuArray(Hy_x);
    Hy_y_gpu = gpuArray(Hy_y);
    Hy_z_gpu = gpuArray(Hy_z);
    
    Hz_x_gpu = gpuArray(Hz_x);
    Hz_y_gpu = gpuArray(Hz_y);
    Hz_z_gpu = gpuArray(Hz_z);
else
    % 无GPU时，绑定CPU数组（修复else语法，确保结构完整）
    warning('未检测到可用GPU，将使用CPU计算！');
    Ex_x_gpu = Ex_x;
    Ex_y_gpu = Ex_y;
    Ex_z_gpu = Ex_z;
    
    Ey_x_gpu = Ey_x;
    Ey_y_gpu = Ey_y;
    Ey_z_gpu = Ey_z;
    
    Ez_x_gpu = Ez_x;
    Ez_y_gpu = Ez_y;
    Ez_z_gpu = Ez_z;
    
    Hx_x_gpu = Hx_x;
    Hx_y_gpu = Hx_y;
    Hx_z_gpu = Hx_z;
    
    Hy_x_gpu = Hy_x;
    Hy_y_gpu = Hy_y;
    Hy_z_gpu = Hy_z;
    
    Hz_x_gpu = Hz_x;
    Hz_y_gpu = Hz_y;
    Hz_z_gpu = Hz_z;
end  % 确保if-else结构有完整的end闭合

% ===== 预计算精确解的Yee网格场值（GPU加速）=====
fprintf('预计算Yee网格上的精确解（M_exact=%d，N_grid=%d）...\n', ...
    M_exact, N_grid);
tic;

% -------- 在GPU上计算精确解（运算全程在GPU，无数据交互）--------
% 精确解E分量
HE_exact_Ex = HE_exact_func(Ex_x_gpu, Ex_y_gpu, Ex_z_gpu, t)';
E_exact_Ex = HE_exact_Ex{4};

HE_exact_Ey = HE_exact_func(Ey_x_gpu, Ey_y_gpu, Ey_z_gpu, t)';
E_exact_Ey = HE_exact_Ey{5};

HE_exact_Ez = HE_exact_func(Ez_x_gpu, Ez_y_gpu, Ez_z_gpu, t)';
E_exact_Ez = HE_exact_Ez{6};

% 精确解H分量
HE_exact_Hx = HE_exact_func(Hx_x_gpu, Hx_y_gpu, Hx_z_gpu, t)';
H_exact_Hx = HE_exact_Hx{1};

HE_exact_Hy = HE_exact_func(Hy_x_gpu, Hy_y_gpu, Hy_z_gpu, t)';
H_exact_Hy = HE_exact_Hy{2};

HE_exact_Hz = HE_exact_func(Hz_x_gpu, Hz_y_gpu, Hz_z_gpu, t)';
H_exact_Hz = HE_exact_Hz{3};

% -------- 将精确解结果转回CPU（便于保存和后续误差计算）--------
if use_gpu
    E_exact_Ex = gather(E_exact_Ex);
    E_exact_Ey = gather(E_exact_Ey);
    E_exact_Ez = gather(E_exact_Ez);
    
    H_exact_Hx = gather(H_exact_Hx);
    H_exact_Hy = gather(H_exact_Hy);
    H_exact_Hz = gather(H_exact_Hz);
end

exact_time = toc;
fprintf('精确解计算完成，用时：%.2f秒\n', exact_time);

% ===== 保存精确解数据到.mat文件 =====
save_filename = sprintf('ex2_exact_HEn_t=%.1f_M=%d_N=%d.mat', ...
    t, M_exact, N_grid); 
save(save_filename, ...
     'E_exact_Ex', 'E_exact_Ey', 'E_exact_Ez', ...  % E场精确解
     'H_exact_Hx', 'H_exact_Hy', 'H_exact_Hz', ...  % H场精确解
     'M_exact', 'N_grid', 'l', 't', 'epsilon', 'mu');% 附加参数
fprintf('精确解已保存至文件：%s\n', save_filename);

%% ====== 循环不同M值，计算E/H场Inf误差与收敛阶（数值解GPU加速）======
fprintf('\n开始计算不同M值的E/H场Inf误差与收敛阶...\n'); % 新增H场说明
fprintf('Yee网格参数：单方向网格数=%d，步长h=%.4f\n', N_grid, h);
fprintf('测试M列表：%s\n\n', num2str(M_list));

total_numer_time = 0; % 统计数值解总耗时
for m_idx = 1:length(M_list)
    M = M_list(m_idx);
    fprintf('正在计算M=%d...\n', M);
    tic;
    
    % 1. 创建当前M对应的数值解函数句柄
    HE_func = create_Maxwell_functions_simple(M, l, epsilon, mu);
    
    % 2. 计算数值解的Yee网格场值（GPU加速）
    % -------- E场数值解（原有）--------
    HE_numer_Ex = HE_func(Ex_x_gpu, Ex_y_gpu, Ex_z_gpu, t)';
    E_numer_Ex = HE_numer_Ex{4};
    
    HE_numer_Ey = HE_func(Ey_x_gpu, Ey_y_gpu, Ey_z_gpu, t)';
    E_numer_Ey = HE_numer_Ey{5};
    
    HE_numer_Ez = HE_func(Ez_x_gpu, Ez_y_gpu, Ez_z_gpu, t)';
    E_numer_Ez = HE_numer_Ez{6};
    
    % -------- H场数值解（新增）--------
    HE_numer_Hx = HE_func(Hx_x_gpu, Hx_y_gpu, Hx_z_gpu, t)';
    H_numer_Hx = HE_numer_Hx{1};
    
    HE_numer_Hy = HE_func(Hy_x_gpu, Hy_y_gpu, Hy_z_gpu, t)';
    H_numer_Hy = HE_numer_Hy{2};
    
    HE_numer_Hz = HE_func(Hz_x_gpu, Hz_y_gpu, Hz_z_gpu, t)';
    H_numer_Hz = HE_numer_Hz{3};
    
    % 3. 将数值解转回CPU（与精确解对比计算误差）
    if use_gpu
        E_numer_Ex = gather(E_numer_Ex);
        E_numer_Ey = gather(E_numer_Ey);
        E_numer_Ez = gather(E_numer_Ez);
        
        H_numer_Hx = gather(H_numer_Hx); % 新增
        H_numer_Hy = gather(H_numer_Hy); % 新增
        H_numer_Hz = gather(H_numer_Hz); % 新增
    end
    
    % 4. 计算E/H场无穷范数误差（Inf）
    % -------- E场误差（原有）--------
    err_Ex = E_exact_Ex - E_numer_Ex;
    err_Ey = E_exact_Ey - E_numer_Ey;
    err_Ez = E_exact_Ez - E_numer_Ez;
    err_E_Inf(m_idx) = max([max(abs(err_Ex(:))), ...
        max(abs(err_Ey(:))), max(abs(err_Ez(:)))]);
    
    % -------- H场误差（新增）--------
    err_Hx = H_exact_Hx - H_numer_Hx;
    err_Hy = H_exact_Hy - H_numer_Hy;
    err_Hz = H_exact_Hz - H_numer_Hz;
    err_H_Inf(m_idx) = max([max(abs(err_Hx(:))), ...
        max(abs(err_Hy(:))), max(abs(err_Hz(:)))]);
    
    % 5. 计算E/H场收敛阶（基于Inf误差）
    if m_idx >= 2
        M_last = M_list(m_idx-1);
        % E场收敛阶（原有）
        order_E_Inf(m_idx) = ...
            log(err_E_Inf(m_idx-1)/err_E_Inf(m_idx)) / log(M/M_last);
        % H场收敛阶（新增）
        order_H_Inf(m_idx) = ...
            log(err_H_Inf(m_idx-1)/err_H_Inf(m_idx)) / log(M/M_last);
    else
        order_E_Inf(m_idx) = NaN;  % 第一个M无收敛阶
        order_H_Inf(m_idx) = NaN;  % 新增
    end
    
    calc_time = toc;
    total_numer_time = total_numer_time + calc_time;
    % 打印结果（新增H场误差）
    fprintf('M=%d计算完成，用时：%.2f秒 | E_inf误差=%.3e | H_inf误差=%.3e\n', ...
        M, calc_time, err_E_Inf(m_idx), err_H_Inf(m_idx));
end

% 释放GPU显存（可选）
if use_gpu && ~isempty(gpu_dev)
    reset(gpu_dev);
    fprintf('\n已释放GPU显存，总数值解耗时：%.2f秒\n', total_numer_time);
end

%% ============= 结果整理与表格输出（扩展E/H场）====================
fprintf('\n=========================================================\n');
fprintf('            Yee网格上Fourier级数E/H场Inf误差与收敛阶\n'); 
fprintf('===========================================================\n');
fprintf('Yee网格参数：单方向网格数=%d，步长h=%.4f\n', N_grid, h);
fprintf('区域范围：[0,%.1f]^3，计算时间t=%.1fs\n\n', l, t);

result_data = [M_list', err_E_Inf, order_E_Inf, err_H_Inf, order_H_Inf];
% 格式化字符串
result_cell = cell(length(M_list), 5); 
for i = 1:length(M_list)
    result_cell{i,1} = num2str(result_data(i,1));          % M值
    result_cell{i,2} = sprintf('%.3e', result_data(i,2));  % E_inf误差
    % E场收敛阶
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

% 创建表格（新增H场列名）
T = cell2table(result_cell, ...
    'VariableNames', ...
    {'M', 'E_Inf误差', 'E收敛阶(Inf)', 'H_Inf误差', 'H收敛阶(Inf)'});
disp(T);
% ===== 保存收敛速度数据到.mat文件 =====
save_filename = sprintf('table_t=%.1f_N=%d_Order.mat', t, N_grid); 
save(save_filename, 'T');% 附加参数
fprintf('收敛已保存至文件');
%% ====================== 收敛阶可视化 ============================
% 1. E/H场Inf误差衰减曲线（对数坐标）
figure('Color','white','Position',[100,100,700,400]);
loglog(M_list, err_E_Inf, 'ro-', 'LineWidth', ...
    1.5, 'MarkerSize', 6, 'DisplayName', 'E场');
hold on; grid on;
loglog(M_list, err_H_Inf, 'bs-', 'LineWidth', ...
    1.5, 'MarkerSize', 6, 'DisplayName', 'H场');
xlabel('Fourier截断数 M');
ylabel('Inf范数误差（对数坐标）');
title(sprintf('E/H场Inf误差衰减曲线（N_grid=%d，GPU加速）', N_grid));
legend('Location','best');
set(gca, 'FontSize', 10);

% 2. E/H场收敛阶变化
figure('Color','white','Position',[100,100,700,400]);
plot(M_list(2:end), order_E_Inf(2:end), ...
    'ro-', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'E');
hold on; grid on;
plot(M_list(2:end), order_H_Inf(2:end), ...
    'bs-', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'H'); 
xlabel('M,N,K');
ylabel('Order');
legend('Location','best');
set(gca, 'FontSize', 10);

fprintf('\n============================================================\n');
fprintf('注：1. 收敛阶基于E/H场Inf范数计算，适配FDTD对比；\n');
fprintf('    2. 精确解+数值解均采用GPU加速，N_grid=128时加速效果显著；\n');
fprintf('    3. 对数坐标下误差曲线斜率绝对值≈收敛阶；\n');
fprintf('    4. E/H场收敛阶计算逻辑一致，均基于Fourier截断数的对数比；\n');
fprintf('    5. 已自动释放GPU显存，避免显存泄漏。\n');
fprintf('============================================================\n');


function [E, H] = t_create_Maxwell_f_exact()
% 用于测试代码的正确性    
% 创建E和H的解析函数
    E1 = @(x, y, z, t)(cos(2*sqrt(3)*pi*t)*cos(2*pi*x)*sin(2*pi*y)*sin(2*pi*z));
    E2 = @(x, y, z, t)-2*(cos(2*sqrt(3)*pi*t)*sin(2*pi*x)*cos(2*pi*y)*sin(2*pi*z));
    E3 = @(x, y, z, t)(cos(2*sqrt(3)*pi*t)*sin(2*pi*x)*sin(2*pi*y)*cos(2*pi*z));
    H1 = @(x, y, z, t)(-sqrt(3)*sin(2*sqrt(3)*pi*t)*sin(2*pi*x)*cos(2*pi*y)*cos(2*pi*z));
    H2 = @(x, y, z, t)0;
    H3 = @(x, y, z, t)(sqrt(3)*sin(2*sqrt(3)*pi*t)*cos(2*pi*x)*cos(2*pi*y)*sin(2*pi*z));
    E = {E1,E2,E3};
    H = {H1,H2,H3};
end
%% ====================== A矩阵生成函数 ===========================
function A = create_A_matrix(m, n, k, mu, epsilon, l)
% CREATE_A_MATRIX 生成Maxwell方程Fourier展开后的6×6系数矩阵A
% 输入参数：
%   m, n, k  - 傅里叶级数的三个整数阶数（可为0或非零）
%   mu       - 介质的磁导率（标量）
%   epsilon  - 介质的介电常数（标量）
%   l        - 空间域边长（标量）
% 输出参数：
%   A        - 6×6的复数矩阵，对应Maxwell方程展开后的系数矩阵

% 1. 初始化6×6零矩阵（复数类型，避免实数截断）
A = zeros(6, 6);

% 2. 定义重复出现的系数，简化代码（2π/l）
coeff = 2 * pi / l;

% 3. 填充第一行非零元素
A(1, 5) = 1i * coeff * k / mu;       % 第一行第五列：(i/μ)*(2πk/l)
A(1, 6) = -1i * coeff * n / mu;      % 第一行第六列：-(i/μ)*(2πn/l)

% 4. 填充第二行非零元素
A(2, 4) = -1i * coeff * k / mu;      % 第二行第四列：-(i/μ)*(2πk/l)
A(2, 6) = 1i * coeff * m / mu;       % 第二行第六列：(i/μ)*(2πm/l)

% 5. 填充第三行非零元素
A(3, 4) = 1i * coeff * n / mu;       % 第三行第四列：(i/μ)*(2πn/l)
A(3, 5) = -1i * coeff * m / mu;      % 第三行第五列：-(i/μ)*(2πm/l)

% 6. 填充第四行非零元素
A(4, 2) = -1i * coeff * k / epsilon; % 第四行第二列：-(i/ε)*(2πk/l)
A(4, 3) = 1i * coeff * n / epsilon;  % 第四行第三列：(i/ε)*(2πn/l)

% 7. 填充第五行非零元素
A(5, 1) = 1i * coeff * k / epsilon;  % 第五行第一列：(i/ε)*(2πk/l)
A(5, 3) = -1i * coeff * m / epsilon; % 第五行第三列：-(i/ε)*(2πm/l)

% 8. 填充第六行非零元素
A(6, 1) = -1i * coeff * n / epsilon; % 第六行第一列：-(i/ε)*(2πn/l)
A(6, 2) = 1i * coeff * m / epsilon;  % 第六行第二列：(i/ε)*(2πm/l)

end

%% ======================= 简化预计算函数 =========================
function modes = precompute_modes_simple(M)
    % 预计算模式的m/n/k和初始X0，移除特征分解相关逻辑
    m_list = -M:M;
    T_all = zeros(size(m_list));
    S_all = zeros(size(m_list));
    for idx = 1:length(m_list)
        m = m_list(idx);
        T_all(idx) = compute_T(m);
        S_all(idx) = compute_S(m);
    end
    
    
    % 步骤2：预分配modes内存（仅存储m/n/k和初始X0）
    % === 核心修改：仅保留m/n/k和X0，移除V/omega/c ===
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
                
                % 计算初始系数
                E1_coeff = -T_m * S_n * S_k;
                E2_coeff = 2 * S_m * T_n * S_k;
                E3_coeff = -S_m * S_n * T_k;
                
                tol = 1e-12;  % 极小阈值
                if abs(E1_coeff) < tol && abs(E2_coeff) < tol && abs(E3_coeff) < tol
                    continue;
                end
                mode_idx = mode_idx + 1;
                X0 = [0; 0; 0; E1_coeff; E2_coeff; E3_coeff];
                
                % 存储模式信息（仅保留m/n/k和X0）
                modes(mode_idx).m = m;
                modes(mode_idx).n = n;
                modes(mode_idx).k = k;
                modes(mode_idx).X0 = X0;
            end
        end
    end
end

%% ====================== 重构场值计算函数 ========================
function HE_func = evaluate_field_simple(x, y, z, t, modes, l, mu, epsilon)
    % 前置验证：x/y/z维度必须一致
    if ~isequal(size(x), size(y), size(z))
        error('x、y、z的维度必须完全一致！当前x维度：%s，y维度：%s，z维度：%s', ...
            mat2str(size(x)), mat2str(size(y)), mat2str(size(z)));
    end

    sz = size(x);
    
    % 初始化场值
    Ex = zeros(sz); Ey = zeros(sz); Ez = zeros(sz);
    Hx = zeros(sz); Hy = zeros(sz); Hz = zeros(sz);
    
    % 遍历每个模式计算贡献
    num_modes = length(modes);
    coeff_2pi_l = 2 * pi / l; % 预计算2π/l
    for mode_idx = 1:num_modes
        % 提取模式参数
        m = modes(mode_idx).m;
        n = modes(mode_idx).n;
        k = modes(mode_idx).k;
        X0 = modes(mode_idx).X0;
       
        % 生成A矩阵 ===
        A = create_A_matrix(m, n, k, mu, epsilon, l);
        % 计算指数矩阵expm(A*t) ===
        expA = expm(A *  t);
        % === 核心修改3：计算Xt = expA * X0 ===
        Xt = expA * X0;
        % 尝试使用特征值分解求解
%         [V, D] = eig(A);
%         V_inv = V \ eye(6);
%         exp_Dt = diag(exp(diag(D) * t));
%         Xt = V * exp_Dt * V_inv * X0;
        % 计算相位因子：exp(1i*2π(mx + ny + kz)/l)
        phase = exp(1i * coeff_2pi_l * (m*x + n*y + k*z));
        
        % 提取场分量并叠加（取实部，因为物理场是实数）
        Hx = Hx + real(Xt(1) * phase);
        Hy = Hy + real(Xt(2) * phase);
        Hz = Hz + real(Xt(3) * phase);
        Ex = Ex + real(Xt(4) * phase);
        Ey = Ey + real(Xt(5) * phase);
        Ez = Ez + real(Xt(6) * phase);
    end
    
    % 整理输出格式
    HE_func = {Hx, Hy, Hz, Ex, Ey, Ez};
end

%% ========================== 辅助函数 ===========================
function HE_func = create_Maxwell_functions_simple(M, l, epsilon, mu)
    % 创建E和H的函数句柄
    modes = precompute_modes_simple(M);
    HE_func= @(x, y, z, t) evaluate_field_simple(x, y, z, t, modes, l, mu, epsilon);
end

%% 计算函数 f(x) =exp(cos(2πx))在[0,1]的 Fourier 系数 T(m)
function T = compute_T(m)
    T = besseli(m, 1); 
end
%% 计算函数 g(x) = sin(2πx)exp(cos(2πx))在[0,1]的 Fourier 系数 S(m)
function S = compute_S(m)
    % 解析简化公式：S(m) = -j * m * I_m(1)
    % I_m(1) = besseli(m, 1)（m阶第一类修正贝塞尔函数）
    if m == 0
        S = 0; 
    else
        S = -1i * m * besseli(m, 1);
    end
end
%% 测试算法正确性
% 计算函数 cos(2πx)、sin(2πx)在[0,1]的 Fourier 系数 T(m)、S(m)
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