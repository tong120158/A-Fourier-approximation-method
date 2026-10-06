clear; close all;

%% Parameters
l = 1;          
epsilon = 1;    
mu = 1;         
c = 1/sqrt(mu*epsilon); 
t = 0.1; 
M_exact = 50;   
% Convergence test settings
N_grid_list = [8, 16, 32, 64, 128]; % grid sizes to test
err_FDTD_E_Inf = zeros(length(N_grid_list), 1); 
order_FDTD_E   = zeros(length(N_grid_list), 1); 
err_FDTD_H_Inf = zeros(length(N_grid_list), 1); 
order_FDTD_H   = zeros(length(N_grid_list), 1); 

%% Convergence loop over different grid sizes
fprintf('Computing FDTD convergence over different grid sizes...\n');
for n_idx = 1:length(N_grid_list)
    N = N_grid_list(n_idx);
    h = l/N;  
    dt = h/(4*c); % CFL condition: dt <= h/c

    % Load exact solution (Yee grid) for the current N
    exact_filename = sprintf('ex2_exact_HEn_t=%.1f_M=%d_N=%d.mat', t, M_exact, N);
    if ~exist(exact_filename, 'file')
        error('Exact solution file not found: %s. Run the Fourier exact-solution code first.', exact_filename);
    end
    load(exact_filename); 

    grid = init_grid(N, N, N, l);
    [E, H] = init_fields(grid);

    t_current = 0;
    fprintf('FDTD N=%d, h=%.4f, dt=%.4f, target t=%.1f...\n', N, h, dt, t);
    tic;
    while t_current < t
        if t_current + dt > t
            dt = t - t_current;
        end

        [E, H] = adi_fdtd(E, H, grid, dt, epsilon, mu);

        t_current = t_current + dt;
    end
    fdtd_time = toc;

    % Component-wise errors vs. the exact solution
    err_Ex = E_exact_Ex(:,:,:) - E.Ex(:,:,:);
    err_Ey = E_exact_Ey(:,:,:) - E.Ey(:,:,:);
    err_Ez = E_exact_Ez(:,:,:) - E.Ez(:,:,:);
    err_Hx = H_exact_Hx(:,:,:) - H.Hx(:,:,:);
    err_Hy = H_exact_Hy(:,:,:) - H.Hy(:,:,:);
    err_Hz = H_exact_Hz(:,:,:) - H.Hz(:,:,:);
    % Inf-norm (max absolute value) of the error
    err_FDTD_E_Inf(n_idx) = max([max(abs(err_Ex(:))), max(abs(err_Ey(:))), max(abs(err_Ez(:)))]);
    err_FDTD_H_Inf(n_idx) = max([max(abs(err_Hx(:))), max(abs(err_Hy(:))), max(abs(err_Hz(:)))]);

    % Convergence order based on h = 1/N
    if n_idx >= 2
        N_prev = N_grid_list(n_idx-1);
        h_prev = l/N_prev;
        h_curr = l/N;
        order_FDTD_E(n_idx) = log(err_FDTD_E_Inf(n_idx-1)/err_FDTD_E_Inf(n_idx)) / log(h_prev/h_curr);
        order_FDTD_H(n_idx) = log(err_FDTD_H_Inf(n_idx-1)/err_FDTD_H_Inf(n_idx)) / log(h_prev/h_curr);
    else
        order_FDTD_E(n_idx) = NaN; 
        order_FDTD_H(n_idx) = NaN;
    end

    fprintf('FDTD N=%d done, time: %.2fs | E_inf=%.3e (order=%.3f) | H_inf=%.3e (order=%.3f)\n', ...
        N, fdtd_time, err_FDTD_E_Inf(n_idx), order_FDTD_E(n_idx), ...
        err_FDTD_H_Inf(n_idx), order_FDTD_H(n_idx));
end

%% Results summary
fprintf('\n============================================================\n');
fprintf('              FDTD Convergence Summary (E/H fields)\n');
fprintf('============================================================\n');
fprintf('Target time: t=%.1fs, wave speed c=%.1f\n\n', t, c);

result_data = [N_grid_list', (l./N_grid_list)', err_FDTD_E_Inf, order_FDTD_E, err_FDTD_H_Inf, order_FDTD_H];
result_cell = cell(length(N_grid_list), 6);
for i = 1:length(N_grid_list)
    result_cell{i,1} = num2str(result_data(i,1));          
    result_cell{i,2} = sprintf('%.4f', result_data(i,2));  
    result_cell{i,3} = sprintf('%.3e', result_data(i,3));  
    if isnan(result_data(i,4))
        result_cell{i,4} = '--';
    else
        result_cell{i,4} = sprintf('%.3f', result_data(i,4)); 
    end
    result_cell{i,5} = sprintf('%.3e', result_data(i,5));  
    if isnan(result_data(i,6))
        result_cell{i,6} = '--';
    else
        result_cell{i,6} = sprintf('%.3f', result_data(i,6)); 
    end
end

T = cell2table(result_cell, ...
    'VariableNames', {'N_grid', 'Grid_spacing_h', 'E_Inf_error', 'E_convergence_order', 'H_Inf_error', 'H_convergence_order'});
disp(T);

%% Convergence visualization (E and H fields)
figure('Color','white','Position',[100,100,700,400]);
loglog(N_grid_list, err_FDTD_E_Inf, 'bo-', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'E field');
hold on; grid on;
loglog(N_grid_list, err_FDTD_H_Inf, 'rs-', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'H field');
xlabel('Yee grid size N_{grid}');
ylabel('Inf-norm error (log-log)');
title('FDTD E/H Inf-norm error vs. grid size');
legend('Location','best');
set(gca, 'FontSize', 10);

figure('Color','white','Position',[100,100,700,400]);
plot(N_grid_list(2:end), order_FDTD_E(2:end), 'bo-', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'E field');
hold on; grid on;
plot(N_grid_list(2:end), order_FDTD_H(2:end), 'rs-', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'H field');
yline(2, 'k--', 'LineWidth', 1, 'DisplayName', '2nd-order reference');
xlabel('Yee grid size N_{grid}');
ylabel('Convergence order (Inf-norm)');
title('FDTD E/H convergence order vs. grid size');
legend('Location','best');
set(gca, 'FontSize', 10);

fprintf('============================================================\n');

%% Grid and parameters
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


%% Initial fields
function [E,H] = init_fields(grid)

Nx=grid.Nx; Ny=grid.Ny; Nz=grid.Nz;
x=grid.x; y=grid.y; z=grid.z;
dx = grid.dx; dy = grid.dy;dz = grid.dz;

% Electric field (Yee staggered)
E.Ex = zeros(Nx,Ny+1,Nz+1);
E.Ey = zeros(Nx+1,Ny,Nz+1);
E.Ez = zeros(Nx+1,Ny+1,Nz);

for i=1:Nx
for j=1:Ny
for k=1:Nz
    xc = x(i,j,k); yc = y(i,j,k); zc = z(i,j,k);
    xc1 = x(i,j,k)+dx/2; yc1 = y(i,j,k)+dy/2; zc1 = z(i,j,k)+dz/2;

    E.Ex(i,j,k) = -exp(cos(2*pi*xc1)) .* sin(2*pi*yc) .* ...
        exp(cos(2*pi*yc)) .* sin(2*pi*zc) .* exp(cos(2*pi*zc));
    E.Ey(i,j,k) = 2*sin(2*pi*xc) .* exp(cos(2*pi*xc)) .* ...
        exp(cos(2*pi*yc1)) .* sin(2*pi*zc) .* exp(cos(2*pi*zc));
    E.Ez(i,j,k) = -sin(2*pi*xc) .* exp(cos(2*pi*xc)) .* ...
        sin(2*pi*yc) .* exp(cos(2*pi*yc)) .* exp(cos(2*pi*zc1));

end
end
end

% Magnetic field (zero initial condition)
H.Hx = zeros(Nx+1,Ny,Nz);
H.Hy = zeros(Nx,Ny+1,Nz);
H.Hz = zeros(Nx,Ny,Nz+1);
end


%% ADI-FDTD update
function [E,H] = adi_fdtd(E,H,grid,dt,eps,mu)

Nx = grid.Nx; Ny=grid.Ny; Nz=grid.Nz;
dx=grid.dx;dy=grid.dy;dz=grid.dz;

alpha = dt/(2*mu);
beta  = dt/(2*eps);

% Snapshot at t = n
Ex1 = E.Ex; Ey1 = E.Ey; Ez1 = E.Ez;
Hx1 = H.Hx; Hy1 = H.Hy; Hz1 = H.Hz;

% Solve Ex, Hy
for i = 1:Nx
for j = 2:Ny
    A = tridiag1(-alpha*beta/(dz^2),(1+2*alpha*beta/(dz^2)),...
        -alpha*beta/(dz^2),Nz-1);
    F = Ex1(i,j,2:Nz)+(beta/dy)*(Hz1(i,j,2:Nz)-Hz1(i,j-1,2:Nz))...
        -(beta/dz)*(Hy1(i,j,2:Nz)-Hy1(i,j,1:Nz-1))...
        -(alpha*beta/(dx*dz))*(Ez1(i+1,j,2:Nz)-Ez1(i,j,2:Nz)...
        -Ez1(i+1,j,1:Nz-1)+Ez1(i,j,1:Nz-1));
    % Boundary conditions
    F(1) = F(1)+alpha*beta*Ex1(i,j,1)/(dz^2);
    F(end)=F(end)+alpha*beta*Ex1(i,j,end)/(dz^2);
    F = reshape(F, Nz-1, 1);
    E.Ex(i,j,2:Nz) = A\F;
    H.Hy(i,j,1:Nz) = Hy1(i,j,1:Nz)+(alpha/dx)*(Ez1(i+1,j,1:Nz)...
        -Ez1(i,j,1:Nz)) - (alpha/dz)*(E.Ex(i,j,2:Nz+1)-E.Ex(i,j,1:Nz));
end
end

% Solve Ey, Hz
for j = 1:Ny
for k = 2:Nz
    A = tridiag1(-alpha*beta/(dx^2),(1+2*alpha*beta/(dx^2)),...
        -alpha*beta/(dx^2),Nx-1);
    F = Ey1(2:Nx,j,k)+(beta/dz)*(Hx1(2:Nx,j,k)-Hx1(2:Nx,j,k-1))...
        -(beta/dx)*(Hz1(2:Nx,j,k)-Hz1(1:Nx-1,j,k))...
        -(alpha*beta/(dx*dy))*(Ex1(2:Nx,j+1,k)-Ex1(2:Nx,j,k)...
        -Ex1(1:Nx-1,j+1,k)+Ex1(1:Nx-1,j,k));
    % Boundary conditions
    F(1) = F(1)+alpha*beta*Ey1(1,j,k)/(dx^2);
    F(end)=F(end)+alpha*beta*Ey1(end,j,k)/(dx^2);
    F = reshape(F, Nx-1, 1);
    E.Ey(2:Nx,j,k) = A\F;
    H.Hz(1:Nx,j,k) = Hz1(1:Nx,j,k)+(alpha/dy)*(Ex1(1:Nx,j+1,k)...
        -Ex1(1:Nx,j,k)) - (alpha/dx)*(E.Ey(2:Nx+1,j,k)-E.Ey(1:Nx,j,k));
end
end

% Solve Ez, Hx
for i = 2:Nx
for k = 1:Nz
    A = tridiag1(-alpha*beta/(dy^2),(1+2*alpha*beta/(dy^2)),...
        -alpha*beta/(dy^2),Ny-1);
    F = Ez1(i,2:Ny,k)+(beta/dx)*(Hy1(i,2:Ny,k)-Hy1(i-1,2:Ny,k))...
        -(beta/dy)*(Hx1(i,2:Ny,k)-Hx1(i,1:Ny-1,k))...
        -(alpha*beta/(dy*dz))*(Ey1(i,2:Ny,k+1)-Ey1(i,2:Ny,k)...
        -Ey1(i,1:Ny-1,k+1)+Ey1(i,1:Ny-1,k));
    % Boundary conditions
    F(1) = F(1)+alpha*beta*Ez1(i,1,k)/(dy^2);
    F(end)=F(end)+alpha*beta*Ez1(i,end,k)/(dy^2);
    F = reshape(F, Ny-1, 1);
    E.Ez(i,2:Ny,k) = A\F;
    H.Hx(i,1:Ny,k) = Hx1(i,1:Ny,k)+(alpha/dz)*(Ey1(i,1:Ny,k+1)...
        -Ey1(i,1:Ny,k)) - (alpha/dy)*(E.Ez(i,2:Ny+1,k)-E.Ez(i,1:Ny,k));
end
end

% Snapshot at t = n + 1/2
Ex1 = E.Ex; Ey1 = E.Ey; Ez1 = E.Ez;
Hx1 = H.Hx; Hy1 = H.Hy; Hz1 = H.Hz;
%--------------------
% Solve Ex, Hz
for i = 1:Nx
for k = 2:Nz
    A = tridiag1(-alpha*beta/(dy^2),(1+2*alpha*beta/(dy^2)),...
        -alpha*beta/(dy^2),Ny-1);
    F = Ex1(i,2:Ny,k)+(beta/dy)*(Hz1(i,2:Ny,k)-Hz1(i,1:Ny-1,k))...
        -(beta/dz)*(Hy1(i,2:Ny,k)-Hy1(i,2:Ny,k-1))...
        -(alpha*beta/(dx*dy))*(Ey1(i+1,2:Ny,k)-Ey1(i,2:Ny,k)...
        -Ey1(i+1,1:Ny-1,k)+Ey1(i,1:Ny-1,k));
    % Boundary conditions
    F(1) = F(1)+alpha*beta*Ex1(i,1,k)/(dy^2);
    F(end)=F(end)+alpha*beta*Ex1(i,end,k)/(dy^2);
    F = reshape(F, Ny-1, 1);
    E.Ex(i,2:Ny,k) = A\F;
    H.Hz(i,1:Ny,k) = Hz1(i,1:Ny,k)+(alpha/dy)*(E.Ex(i,2:Ny+1,k)...
        -E.Ex(i,1:Ny,k)) - (alpha/dx)*(Ey1(i+1,1:Ny,k)-Ey1(i,1:Ny,k));
end
end

% Solve Ey, Hx
for i = 2:Nx
for j = 1:Ny
    A = tridiag1(-alpha*beta/(dz^2),(1+2*alpha*beta/(dz^2)),...
        -alpha*beta/(dz^2),Nz-1);
    F = Ey1(i,j,2:Nz)+(beta/dz)*(Hx1(i,j,2:Nz)-Hx1(i,j,1:Nz-1))...
        -(beta/dx)*(Hz1(i,j,2:Nz)-Hz1(i-1,j,2:Nz))...
        -(alpha*beta/(dy*dz))*(Ez1(i,j+1,2:Nz)-Ez1(i,j,2:Nz)...
        -Ez1(i,j+1,1:Nz-1)+Ez1(i,j,1:Nz-1));
    % Boundary conditions
    F(1) = F(1)+alpha*beta*Ey1(i,j,1)/(dz^2);
    F(end)=F(end)+alpha*beta*Ey1(i,j,end)/(dz^2);
    F = reshape(F, Nz-1, 1);
    E.Ey(i,j,2:Nz) = A\F;
    H.Hx(i,j,1:Nz) = Hx1(i,j,1:Nz)+(alpha/dz)*(E.Ey(i,j,2:Nz+1)...
        -E.Ey(i,j,1:Nz)) - (alpha/dy)*(Ez1(i,j+1,1:Nz)-Ez1(i,j,1:Nz));
end
end

% Solve Ez, Hy
for j = 2:Nx
for k = 1:Nz
    A = tridiag1(-alpha*beta/(dx^2),(1+2*alpha*beta/(dx^2)),...
        -alpha*beta/(dx^2),Nx-1);
    F = Ez1(2:Nx,j,k)+(beta/dx)*(Hy1(2:Nx,j,k)-Hy1(1:Nx-1,j,k))...
        -(beta/dy)*(Hx1(2:Nx,j,k)-Hx1(2:Nx,j-1,k))...
        -(alpha*beta/(dx*dz))*(Ex1(2:Nx,j,k+1)-Ex1(2:Nx,j,k)...
        -Ex1(1:Nx-1,j,k+1)+Ex1(1:Nx-1,j,k));
    % Boundary conditions
    F(1) = F(1)+alpha*beta*Ez1(1,j,k)/(dx^2);
    F(end)=F(end)+alpha*beta*Ez1(end,j,k)/(dx^2);
    F = reshape(F, Nx-1, 1);
    E.Ez(2:Nx,j,k) = A\F;
    H.Hy(1:Nx,j,k) = Hy1(1:Nx,j,k)+(alpha/dx)*(E.Ez(2:Nx+1,j,k)...
        -E.Ez(1:Nx,j,k)) - (alpha/dz)*(Ex1(1:Nx,j,k+1)-Ex1(1:Nx,j,k));
end
end
end

%% Classical FDTD update (CFL stable)
function [E,H] = fdtd(E,H,grid,dt,eps,mu)

Nx = grid.Nx; Ny=grid.Ny; Nz=grid.Nz;
dx=grid.dx;dy=grid.dy;dz=grid.dz;

% Store old values
Ex_old = E.Ex; Ey_old = E.Ey; Ez_old = E.Ez;
Hx_old = H.Hx; Hy_old = H.Hy; Hz_old = H.Hz;

% Time-stepping parameters
c = 1 / sqrt(eps * mu);
CFL_num = c * dt * sqrt(1/dx^2 + 1/dy^2 + 1/dz^2);

if CFL_num > 1
    warning('CFL number %.4f > 1, simulation may be unstable!', CFL_num);
end

% Update H at n+1/2 (following the Yee staggered indexing)
for i = 2:Nx
for j = 1:Ny
for k = 1:Nz
    % Hx update
    H.Hx(i,j,k) = Hx_old(i,j,k) + dt/(mu) * ...
        ((Ey_old(i,j,k+1) - Ey_old(i,j,k))/dz - ...
         (Ez_old(i,j+1,k) - Ez_old(i,j,k))/dy);
end
end
end

for i = 1:Nx
for j = 2:Ny
for k = 1:Nz
    % Hy update
    H.Hy(i,j,k) = Hy_old(i,j,k) + dt/(mu) * ...
        ((Ez_old(i+1,j,k) - Ez_old(i,j,k))/dx - ...
         (Ex_old(i,j,k+1) - Ex_old(i,j,k))/dz);
end
end
end

for i = 1:Nx
for j = 1:Ny
for k = 2:Nz
    % Hz update
    H.Hz(i,j,k) = Hz_old(i,j,k) + dt/(mu) * ...
        ((Ex_old(i,j+1,k) - Ex_old(i,j,k))/dy - ...
         (Ey_old(i+1,j,k) - Ey_old(i,j,k))/dx);
end
end
end

% Update E at n+1
for i = 1:Nx
for j = 2:Ny
for k = 2:Nz
    % Ex update
    E.Ex(i,j,k) = Ex_old(i,j,k) + dt/(eps) * ...
        ((H.Hz(i,j,k) - H.Hz(i,j-1,k))/dy - ...
         (H.Hy(i,j,k) - H.Hy(i,j,k-1))/dz);
end
end
end

for i = 2:Nx
for j = 1:Ny
for k = 2:Nz
    % Ey update
    E.Ey(i,j,k) = Ey_old(i,j,k) + dt/(eps) * ...
        ((H.Hx(i,j,k) - H.Hx(i,j,k-1))/dz - ...
         (H.Hz(i,j,k) - H.Hz(i-1,j,k))/dx);
end
end
end

for i = 2:Nx
for j = 2:Ny
for k = 1:Nz
    % Ez update
    E.Ez(i,j,k) = Ez_old(i,j,k) + dt/(eps) * ...
        ((H.Hy(i,j,k) - H.Hy(i-1,j,k))/dx - ...
         (H.Hx(i,j,k) - H.Hx(i,j-1,k))/dy);
end
end
end
% Boundary conditions
[E,H] = apply_bc_PBC(E,H);
end

%% Boundary conditions
function [E,H] = apply_bc_PBC(E,H)

E.Ex(:,1,:) = 0;E.Ex(:,end,:) = 0;E.Ex(:,:,1) = 0;E.Ex(:,:,end) = 0;
E.Ey(1,:,:) = 0;E.Ey(end,:,:) = 0;E.Ey(:,:,1) = 0;E.Ey(:,:,end) = 0;
E.Ez(1,:,:) = 0;E.Ez(end,:,:) = 0;E.Ez(:,1,:) = 0;E.Ez(:,end,:) = 0;


H.Hx(1,:,:) = 0; H.Hx(end,:,:) = 0;
H.Hy(:,1,:) = 0; H.Hy(:,end,:) = 0;
H.Hz(:,:,1) = 0; H.Hz(:,:,end) = 0;

end
