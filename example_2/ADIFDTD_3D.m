clear; close all; 

%% ==================== 1. 基础参数设置 ====================
% 物理参数
l = 1;          % 区域边长 [0,l]^3
epsilon = 1;    % 介电常数
mu = 1;         % 磁导率
c = 1/sqrt(mu*epsilon); % 波速
t = 0.1; % 目标计算时间（与精确解一致）
M_exact = 50;   % 精确解的Fourier截断数
% FDTD收敛性测试参数
N_grid_list = [8, 16, 32, 64]; % 测试不同网格数
err_FDTD_E_Inf = zeros(length(N_grid_list), 1); % FDTD的E_inf误差
order_FDTD_E = zeros(length(N_grid_list), 1);   % FDTD的收敛阶
err_FDTD_H_Inf = zeros(length(N_grid_list), 1); % FDTD的E_inf误差
order_FDTD_H = zeros(length(N_grid_list), 1);   % FDTD的收敛阶
%% ==================== 2. 循环不同网格数，计算FDTD误差与收敛阶 ====================
fprintf('开始计算FDTD不同网格数的收敛速度...\n');
for n_idx = 1:length(N_grid_list)
    N = N_grid_list(n_idx);
    h = l/N; % 网格步长
    dt = h/(4*c); % CFL条件：dt ≤ h/c，保证稳定
    
    % -------- 2.1 读取对应N的精确解（Yee网格）--------
    exact_filename = sprintf('ex2_exact_HEn_t=%.1f_M=%d_N=%d.mat', t, M_exact, N);
    if ~exist(exact_filename, 'file')
        error('未找到精确解文件：%s，请先运行Fourier精确解代码生成！', exact_filename);
    end
    load(exact_filename); % 加载：E_exact_Ex/Ey/Ez, H_exact_Hx/Hy/Hz
    
    % -------- 2.2 初始化Yee网格与场量（调用修改后的init_fields）--------
    grid = init_grid(N, N, N, l); % 初始化网格（未修改，省略实现）
    [E, H] = init_fields(grid);   % 关键：初始条件已替换为用户公式
    
    % -------- 2.3 FDTD时间步进（更新到t_target）--------
    t_current = 0;
    fprintf('FDTD N=%d，h=%.4f，dt=%.4f，目标时间t=%.1f...\n', N, h, dt, t);
    tic;
    while t_current < t
        % 限制最后一步时间，避免超目标时间
        if t_current + dt > t
            dt = t - t_current;
        end
        
        % 调用FDTD核心更新函数（未修改，省略实现）
        [E, H] = adi_fdtd(E, H, grid, dt, epsilon, mu);
        
        % 更新当前时间
        t_current = t_current + dt;
    end
    fdtd_time = toc;
    
    % -------- 2.4 计算FDTD与精确解的E_inf误差 --------
    % 计算各分量误差
    err_Ex = E_exact_Ex(:,:,:) - E.Ex(:,:,:);
    err_Ey = E_exact_Ey(:,:,:) - E.Ey(:,:,:);
    err_Ez = E_exact_Ez(:,:,:) - E.Ez(:,:,:);
    err_Hx = H_exact_Hx(:,:,:) - H.Hx(:,:,:);
    err_Hy = H_exact_Hy(:,:,:) - H.Hy(:,:,:);
    err_Hz = H_exact_Hz(:,:,:) - H.Hz(:,:,:);
    % 无穷范数误差（绝对值最大值）
    err_FDTD_E_Inf(n_idx) = max([max(abs(err_Ex(:))), max(abs(err_Ey(:))), max(abs(err_Ez(:)))]);
    err_FDTD_H_Inf(n_idx) = max([max(abs(err_Hx(:))), max(abs(err_Hy(:))), max(abs(err_Hz(:)))]);
    % -------- 2.5 计算FDTD收敛阶（基于网格步长h=1/N）--------
    if n_idx >= 2
        N_prev = N_grid_list(n_idx-1);
        h_prev = l/N_prev;
        h_curr = l/N;
        % FDTD收敛阶公式：p = ln(前一误差/当前误差) / ln(h_prev/h_curr)
        order_FDTD_E(n_idx) = log(err_FDTD_E_Inf(n_idx-1)/err_FDTD_E_Inf(n_idx)) / log(h_prev/h_curr);
        order_FDTD_H(n_idx) = log(err_FDTD_H_Inf(n_idx-1)/err_FDTD_H_Inf(n_idx)) / log(h_prev/h_curr);
    else
        order_FDTD_E(n_idx) = NaN; % 第一个网格数无收敛阶
        order_FDTD_H(n_idx) = NaN;
    end
    
    fprintf('FDTD N=%d 计算完成，用时：%.2f秒 | E_inf误差=%.3e（收敛阶=%.3f）| H_inf误差=%.3e（收敛阶=%.3f）\n', ...
        N, fdtd_time, err_FDTD_E_Inf(n_idx), order_FDTD_E(n_idx), err_FDTD_H_Inf(n_idx), order_FDTD_H(n_idx));
end

%% ==================== 3. 结果整理与输出 ====================
fprintf('\n============================================================\n');
fprintf('                    FDTD收敛速度结果汇总（E/H场）\n'); % 新增注释
fprintf('============================================================\n');
fprintf('目标计算时间：t=%.1fs，波速c=%.1f\n\n', t, c);

% 构造结果数组（新增H场列）
result_data = [N_grid_list', (l./N_grid_list)', err_FDTD_E_Inf, order_FDTD_E, err_FDTD_H_Inf, order_FDTD_H];
% 格式化字符串（扩展为6列）
result_cell = cell(length(N_grid_list), 6);
for i = 1:length(N_grid_list)
    result_cell{i,1} = num2str(result_data(i,1));          % N_grid
    result_cell{i,2} = sprintf('%.4f', result_data(i,2));  % 网格步长h
    result_cell{i,3} = sprintf('%.3e', result_data(i,3));  % E_Inf误差
    % E场收敛阶
    if isnan(result_data(i,4))
        result_cell{i,4} = '--';
    else
        result_cell{i,4} = sprintf('%.3f', result_data(i,4)); 
    end
    result_cell{i,5} = sprintf('%.3e', result_data(i,5));  % H_Inf误差（新增）
    % H场收敛阶（新增）
    if isnan(result_data(i,6))
        result_cell{i,6} = '--';
    else
        result_cell{i,6} = sprintf('%.3f', result_data(i,6)); 
    end
end

% 创建表格（新增H场列名）
T = cell2table(result_cell, ...
    'VariableNames', {'N_grid', '网格步长h', 'E_Inf误差', 'E场收敛阶', 'H_Inf误差', 'H场收敛阶'});
disp(T);

%% ==================== 4. 收敛性可视化（扩展为E/H场） ====================
% 4.1 E/H场Inf误差随网格数的变化（对数坐标）
figure('Color','white','Position',[100,100,700,400]);
loglog(N_grid_list, err_FDTD_E_Inf, 'bo-', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'E场');
hold on; grid on;
loglog(N_grid_list, err_FDTD_H_Inf, 'rs-', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'H场'); % 新增
xlabel('Yee网格数 N_grid');
ylabel('Inf范数误差（对数坐标）');
title('FDTD E/H场Inf误差随网格数的变化');
legend('Location','best');
set(gca, 'FontSize', 10);

% 4.2 E/H场收敛阶变化
figure('Color','white','Position',[100,100,700,400]);
plot(N_grid_list(2:end), order_FDTD_E(2:end), 'bo-', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'E场');
hold on; grid on;
plot(N_grid_list(2:end), order_FDTD_H(2:end), 'rs-', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'H场'); % 新增
% 绘制2阶收敛参考线（FDTD理论收敛阶）
yline(2, 'k--', 'LineWidth', 1, 'DisplayName', '2阶收敛参考线');
xlabel('Yee网格数 N_grid');
ylabel('收敛阶（Inf范数）');
title('FDTD E/H场收敛阶随网格数的变化');
legend('Location','best');
set(gca, 'FontSize', 10);

fprintf('\n============================================================\n');
fprintf('注：1. FDTD收敛阶基于网格步长h=1/N_grid计算，理论值为2阶；\n');
fprintf('    2. 初始条件严格按用户公式赋值，H^0=0；\n');
fprintf('    3. 误差为FDTD E/H场与Fourier精确解的Inf范数误差；\n'); % 新增H场说明
fprintf('    4. H场收敛阶计算逻辑与E场一致，均基于网格步长的对数比。\n'); % 新增
fprintf('============================================================\n');

%% ==================== 5. 修改后的初始条件函数（核心修改）====================



%% 网格与参数
function grid = init_grid(Nx,Ny,Nz,L)

dx = L/Nx; dy = L/Ny; dz = L/Nz;

grid.Nx = Nx; grid.Ny = Ny; grid.Nz = Nz;
grid.dx = dx; grid.dy = dy; grid.dz = dz;

[x,y,z] = ndgrid( ...
    linspace(0,L,Nx+1), ...
    linspace(0,L,Ny+1), ...
    linspace(0,L,Nz+1));

grid.x = x; grid.y = y; grid.z = z;
end


%% 初始条件
function [E,H] = init_fields(grid)

Nx=grid.Nx; Ny=grid.Ny; Nz=grid.Nz;
x=grid.x; y=grid.y; z=grid.z;
dx = grid.dx; dy = grid.dy;dz = grid.dz;
%--------------------
% 电场（Yee）
%--------------------
E.Ex = zeros(Nx,Ny+1,Nz+1);
E.Ey = zeros(Nx+1,Ny,Nz+1);
E.Ez = zeros(Nx+1,Ny+1,Nz);

for i=1:Nx
for j=1:Ny
for k=1:Nz
    xc = x(i,j,k); yc = y(i,j,k); zc = z(i,j,k);
    xc1 = x(i,j,k)+dx/2; yc1 = y(i,j,k)+dy/2; zc1 = z(i,j,k)+dz/2;
  % 例2
    E.Ex(i,j,k) = -exp(cos(2*pi*xc1)) .* sin(2*pi*yc) .* ...
        exp(cos(2*pi*yc)) .* sin(2*pi*zc) .* exp(cos(2*pi*zc));
    E.Ey(i,j,k) = 2*sin(2*pi*xc) .* exp(cos(2*pi*xc)) .* ...
        exp(cos(2*pi*yc1)) .* sin(2*pi*zc) .* exp(cos(2*pi*zc));
    E.Ez(i,j,k) = -sin(2*pi*xc) .* exp(cos(2*pi*xc)) .* ...
        sin(2*pi*yc) .* exp(cos(2*pi*yc)) .* exp(cos(2*pi*zc1));
%   % 例1
%     E.Ex(i,j,k) = cos(pi*xc1)*sin(pi*yc)*sin(pi*zc);
%     E.Ey(i,j,k) = -2*sin(pi*xc)*cos(pi*(yc1))*sin(pi*zc);
%     E.Ez(i,j,k) = sin(pi*xc)*sin(pi*yc)*cos(pi*(zc1));
end
end
end

%--------------------
% 磁场（初始为 0）
%--------------------
H.Hx = zeros(Nx+1,Ny,Nz);
H.Hy = zeros(Nx,Ny+1,Nz);
H.Hz = zeros(Nx,Ny,Nz+1);
end


%% ADI-FDTD更新
function [E,H] = adi_fdtd(E,H,grid,dt,eps,mu)

Nx = grid.Nx; Ny=grid.Ny; Nz=grid.Nz;
dx=grid.dx;dy=grid.dy;dz=grid.dz;

alpha = dt/(2*mu);
beta  = dt/(2*eps);

% 先记录t = n 的信息
Ex1 = E.Ex; Ey1 = E.Ey; Ez1 = E.Ez;
Hx1 = H.Hx; Hy1 = H.Hy; Hz1 = H.Hz;
%--------------------
% 计算Ex,Hy
for i = 1:Nx
for j = 2:Ny
    A = tridiag1(-alpha*beta/(dz^2),(1+2*alpha*beta/(dz^2)),...
        -alpha*beta/(dz^2),Nz-1);
    F = Ex1(i,j,2:Nz)+(beta/dy)*(Hz1(i,j,2:Nz)-Hz1(i,j-1,2:Nz))...
        -(beta/dz)*(Hy1(i,j,2:Nz)-Hy1(i,j,1:Nz-1))...
        -(alpha*beta/(dx*dz))*(Ez1(i+1,j,2:Nz)-Ez1(i,j,2:Nz)...
        -Ez1(i+1,j,1:Nz-1)+Ez1(i,j,1:Nz-1));
    % 代入边界条件
    F(1) = F(1)+alpha*beta*Ex1(i,j,1)/(dz^2);
    F(end)=F(end)+alpha*beta*Ex1(i,j,end)/(dz^2);
    F = reshape(F, Nz-1, 1);
    E.Ex(i,j,2:Nz) = A\F;
    H.Hy(i,j,1:Nz) = Hy1(i,j,1:Nz)+(alpha/dx)*(Ez1(i+1,j,1:Nz)...
        -Ez1(i,j,1:Nz)) - (alpha/dz)*(E.Ex(i,j,2:Nz+1)-E.Ex(i,j,1:Nz));
end
end

% 计算Ey,Hz
for j = 1:Ny
for k = 2:Nz
    A = tridiag1(-alpha*beta/(dx^2),(1+2*alpha*beta/(dx^2)),...
        -alpha*beta/(dx^2),Nx-1);
    F = Ey1(2:Nx,j,k)+(beta/dz)*(Hx1(2:Nx,j,k)-Hx1(2:Nx,j,k-1))...
        -(beta/dx)*(Hz1(2:Nx,j,k)-Hz1(1:Nx-1,j,k))...
        -(alpha*beta/(dx*dy))*(Ex1(2:Nx,j+1,k)-Ex1(2:Nx,j,k)...
        -Ex1(1:Nx-1,j+1,k)+Ex1(1:Nx-1,j,k));
    % 代入边界条件
    F(1) = F(1)+alpha*beta*Ey1(1,j,k)/(dx^2);
    F(end)=F(end)+alpha*beta*Ey1(end,j,k)/(dx^2);
    F = reshape(F, Nx-1, 1);
    E.Ey(2:Nx,j,k) = A\F;
    H.Hz(1:Nx,j,k) = Hz1(1:Nx,j,k)+(alpha/dy)*(Ex1(1:Nx,j+1,k)...
        -Ex1(1:Nx,j,k)) - (alpha/dx)*(E.Ey(2:Nx+1,j,k)-E.Ey(1:Nx,j,k));
end
end

% 计算Ez,Hx
for i = 2:Nx
for k = 1:Nz
    A = tridiag1(-alpha*beta/(dy^2),(1+2*alpha*beta/(dy^2)),...
        -alpha*beta/(dy^2),Ny-1);
    F = Ez1(i,2:Ny,k)+(beta/dx)*(Hy1(i,2:Ny,k)-Hy1(i-1,2:Ny,k))...
        -(beta/dy)*(Hx1(i,2:Ny,k)-Hx1(i,1:Ny-1,k))...
        -(alpha*beta/(dy*dz))*(Ey1(i,2:Ny,k+1)-Ey1(i,2:Ny,k)...
        -Ey1(i,1:Ny-1,k+1)+Ey1(i,1:Ny-1,k));
    % 代入边界条件
    F(1) = F(1)+alpha*beta*Ez1(i,1,k)/(dy^2);
    F(end)=F(end)+alpha*beta*Ez1(i,end,k)/(dy^2);
    F = reshape(F, Ny-1, 1);
    E.Ez(i,2:Ny,k) = A\F;
    H.Hx(i,1:Ny,k) = Hx1(i,1:Ny,k)+(alpha/dz)*(Ey1(i,1:Ny,k+1)...
        -Ey1(i,1:Ny,k)) - (alpha/dy)*(E.Ez(i,2:Ny+1,k)-E.Ez(i,1:Ny,k));
end
end

% 先记录t = n + 1/2的信息
Ex1 = E.Ex; Ey1 = E.Ey; Ez1 = E.Ez;
Hx1 = H.Hx; Hy1 = H.Hy; Hz1 = H.Hz;
%--------------------
% 计算Ex,Hz
for i = 1:Nx
for k = 2:Nz
    A = tridiag1(-alpha*beta/(dy^2),(1+2*alpha*beta/(dy^2)),...
        -alpha*beta/(dy^2),Ny-1);
    F = Ex1(i,2:Ny,k)+(beta/dy)*(Hz1(i,2:Ny,k)-Hz1(i,1:Ny-1,k))...
        -(beta/dz)*(Hy1(i,2:Ny,k)-Hy1(i,2:Ny,k-1))...
        -(alpha*beta/(dx*dy))*(Ey1(i+1,2:Ny,k)-Ey1(i,2:Ny,k)...
        -Ey1(i+1,1:Ny-1,k)+Ey1(i,1:Ny-1,k));
    % 代入边界条件
    F(1) = F(1)+alpha*beta*Ex1(i,1,k)/(dy^2);
    F(end)=F(end)+alpha*beta*Ex1(i,end,k)/(dy^2);
    F = reshape(F, Ny-1, 1);
    E.Ex(i,2:Ny,k) = A\F;
    H.Hz(i,1:Ny,k) = Hz1(i,1:Ny,k)+(alpha/dy)*(E.Ex(i,2:Ny+1,k)...
        -E.Ex(i,1:Ny,k)) - (alpha/dx)*(Ey1(i+1,1:Ny,k)-Ey1(i,1:Ny,k));
end
end

% 计算Ey,Hx
for i = 2:Nx
for j = 1:Ny
    A = tridiag1(-alpha*beta/(dz^2),(1+2*alpha*beta/(dz^2)),...
        -alpha*beta/(dz^2),Nz-1);
    F = Ey1(i,j,2:Nz)+(beta/dz)*(Hx1(i,j,2:Nz)-Hx1(i,j,1:Nz-1))...
        -(beta/dx)*(Hz1(i,j,2:Nz)-Hz1(i-1,j,2:Nz))...
        -(alpha*beta/(dy*dz))*(Ez1(i,j+1,2:Nz)-Ez1(i,j,2:Nz)...
        -Ez1(i,j+1,1:Nz-1)+Ez1(i,j,1:Nz-1));
    % 代入边界条件
    F(1) = F(1)+alpha*beta*Ey1(i,j,1)/(dz^2);
    F(end)=F(end)+alpha*beta*Ey1(i,j,end)/(dz^2);
    F = reshape(F, Nz-1, 1);
    E.Ey(i,j,2:Nz) = A\F;
    H.Hx(i,j,1:Nz) = Hx1(i,j,1:Nz)+(alpha/dz)*(E.Ey(i,j,2:Nz+1)...
        -E.Ey(i,j,1:Nz)) - (alpha/dy)*(Ez1(i,j+1,1:Nz)-Ez1(i,j,1:Nz));
end
end

% 计算Ez,Hy
for j = 2:Nx
for k = 1:Nz
    A = tridiag1(-alpha*beta/(dx^2),(1+2*alpha*beta/(dx^2)),...
        -alpha*beta/(dx^2),Nx-1);
    F = Ez1(2:Nx,j,k)+(beta/dx)*(Hy1(2:Nx,j,k)-Hy1(1:Nx-1,j,k))...
        -(beta/dy)*(Hx1(2:Nx,j,k)-Hx1(2:Nx,j-1,k))...
        -(alpha*beta/(dx*dz))*(Ex1(2:Nx,j,k+1)-Ex1(2:Nx,j,k)...
        -Ex1(1:Nx-1,j,k+1)+Ex1(1:Nx-1,j,k));
    % 代入边界条件
    F(1) = F(1)+alpha*beta*Ez1(1,j,k)/(dx^2);
    F(end)=F(end)+alpha*beta*Ez1(end,j,k)/(dx^2);
    F = reshape(F, Nx-1, 1);
    E.Ez(2:Nx,j,k) = A\F;
    H.Hy(1:Nx,j,k) = Hy1(1:Nx,j,k)+(alpha/dx)*(E.Ez(2:Nx+1,j,k)...
        -E.Ez(1:Nx,j,k)) - (alpha/dz)*(Ex1(1:Nx,j,k+1)-Ex1(1:Nx,j,k));
end
end
end

%% FDTD 经典公式 （时间满足CFL稳定条件）
function [E,H] = fdtd(E,H,grid,dt,eps,mu)

Nx = grid.Nx; Ny=grid.Ny; Nz=grid.Nz;
dx=grid.dx;dy=grid.dy;dz=grid.dz;

% 存储旧值
Ex_old = E.Ex; Ey_old = E.Ey; Ez_old = E.Ez;
Hx_old = H.Hx; Hy_old = H.Hy; Hz_old = H.Hz;

% 时间步进参数
c = 1 / sqrt(eps * mu);
CFL_num = c * dt * sqrt(1/dx^2 + 1/dy^2 + 1/dz^2);

if CFL_num > 1
    warning('CFL数 %.4f > 1，仿真可能不稳定！', CFL_num);
end

% 更新磁场 (n+1/2 时刻)
% 注意：这里需要根据Yee网格的索引规则正确实现
for i = 2:Nx
for j = 1:Ny
for k = 1:Nz
    % Hx更新
    H.Hx(i,j,k) = Hx_old(i,j,k) + dt/(mu) * ...
        ((Ey_old(i,j,k+1) - Ey_old(i,j,k))/dz - ...
         (Ez_old(i,j+1,k) - Ez_old(i,j,k))/dy);
end
end
end

for i = 1:Nx
for j = 2:Ny
for k = 1:Nz
    % Hy更新
    H.Hy(i,j,k) = Hy_old(i,j,k) + dt/(mu) * ...
        ((Ez_old(i+1,j,k) - Ez_old(i,j,k))/dx - ...
         (Ex_old(i,j,k+1) - Ex_old(i,j,k))/dz);
end
end
end

for i = 1:Nx
for j = 1:Ny
for k = 2:Nz
    % Hz更新
    H.Hz(i,j,k) = Hz_old(i,j,k) + dt/(mu) * ...
        ((Ex_old(i,j+1,k) - Ex_old(i,j,k))/dy - ...
         (Ey_old(i+1,j,k) - Ey_old(i,j,k))/dx);
end
end
end

% 更新电场 (n+1 时刻)
for i = 1:Nx
for j = 2:Ny
for k = 2:Nz
    % Ex更新
    E.Ex(i,j,k) = Ex_old(i,j,k) + dt/(eps) * ...
        ((H.Hz(i,j,k) - H.Hz(i,j-1,k))/dy - ...
         (H.Hy(i,j,k) - H.Hy(i,j,k-1))/dz);
end
end
end

for i = 2:Nx
for j = 1:Ny
for k = 2:Nz
    % Ey更新
    E.Ey(i,j,k) = Ey_old(i,j,k) + dt/(eps) * ...
        ((H.Hx(i,j,k) - H.Hx(i,j,k-1))/dz - ...
         (H.Hz(i,j,k) - H.Hz(i-1,j,k))/dx);
end
end
end

for i = 2:Nx
for j = 2:Ny
for k = 1:Nz
    % Ez更新
    E.Ez(i,j,k) = Ez_old(i,j,k) + dt/(eps) * ...
        ((H.Hy(i,j,k) - H.Hy(i-1,j,k))/dx - ...
         (H.Hx(i,j,k) - H.Hx(i,j-1,k))/dy);
end
end
end
% 边界条件
[E,H] = apply_bc_PBC(E,H);
end

%% 边界条件
function [E,H] = apply_bc_PBC(E,H)

E.Ex(:,1,:) = 0;E.Ex(:,end,:) = 0;E.Ex(:,:,1) = 0;E.Ex(:,:,end) = 0;
E.Ey(1,:,:) = 0;E.Ey(end,:,:) = 0;E.Ey(:,:,1) = 0;E.Ey(:,:,end) = 0;
E.Ez(1,:,:) = 0;E.Ez(end,:,:) = 0;E.Ez(:,1,:) = 0;E.Ez(:,end,:) = 0;


H.Hx(1,:,:) = 0; H.Hx(end,:,:) = 0;
H.Hy(:,1,:) = 0; H.Hy(:,end,:) = 0;
H.Hz(:,:,1) = 0; H.Hz(:,:,end) = 0;

end





