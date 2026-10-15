%% ================================================================
%  Simulation figure export:
%  1) Ideal scatterer map: equal-amplitude position display, no colorbar
%  2) Full-aperture RD image: original RD processing, with colorbar
%
%  保存文件：
%  out_sim_figs/sim_ideal_scatterer_equal_no_colorbar.eps
%  out_sim_figs/sim_ideal_scatterer_equal_no_colorbar.png
%  out_sim_figs/sim_full_aperture_RD_original.eps
%  out_sim_figs/sim_full_aperture_RD_original.png
%  out_sim_figs/sim_scatterer_RD_original_data.mat
% ================================================================
clear; clc; close all;

fprintf('=== Simulation: equal scatterer display + original RD output ===\n');

%% ================= Step 1: Radar Parameters =================
radar = struct();

radar.c   = 3e8;
radar.fc  = 10e9;        % Hz
radar.Bw  = 600e6;       % Hz
radar.PRF = 100;         % Hz

radar.Nr  = 256;         % range cells
radar.Na  = 256;         % azimuth pulses

radar.lambda  = radar.c / radar.fc;
radar.delta_r = radar.c / (2 * radar.Bw);

% constant angular velocity
radar.omega_r = 0.06;    % rad/s

% fast-time grid after pulse compression
radar.t_fast = ((0:radar.Nr-1) - radar.Nr/2) / radar.Bw;

% slow-time grid
radar.tm = ((0:radar.Na-1) - radar.Na/2) / radar.PRF;

fprintf('\nRadar parameters:\n');
fprintf('  fc       = %.2f GHz\n', radar.fc / 1e9);
fprintf('  Bw       = %.1f MHz\n', radar.Bw / 1e6);
fprintf('  PRF      = %.1f Hz\n', radar.PRF);
fprintf('  Nr x Na  = %d x %d\n', radar.Nr, radar.Na);
fprintf('  delta_r  = %.4f m\n', radar.delta_r);

%% ================= Step 2: Generate Aircraft-Like Scatterers =================
scene = struct();

scene.numBlocks   = 30;
scene.ptsPerBlock = 3;

% 一个图像 bin 对应的物理尺寸
scene.dy_bin = radar.c / (2 * radar.Bw);
scene.dx_bin = radar.lambda * radar.PRF / (2 * radar.omega_r * radar.Na);

% 保持你原来的 block 尺度，不要改成 1.2 或完全落栅格
scene.block_dx = 0.8 * scene.dx_bin;
scene.block_dy = 0.8 * scene.dy_bin;

% 飞机轮廓参数
scene.body_len   = 14;
scene.wing_span  = 18;
scene.tail_span  = 7;
scene.wing_y     = 2.5;
scene.tail_y     = -5.0;

% 保持原来的轻微吸附和轻微扰动
% 注意：这里不要改成 snap_ratio=1, jitter=0，否则 RD 会变成过于理想的点图
scene.snap_to_grid  = false;
scene.snap_ratio_x  = 0.85;
scene.snap_ratio_y  = 0.85;
scene.jitter_center = 0.08;
scene.jitter_point  = 0.05;

scatterers = generate_aircraft_like_physical_scatterers(scene);

fprintf('\nScatterers generated:\n');
fprintf('  blocks      = %d\n', scene.numBlocks);
fprintf('  scatterers  = %d\n', numel(scatterers));
fprintf('  dx_bin      = %.4f m\n', scene.dx_bin);
fprintf('  dy_bin      = %.4f m\n', scene.dy_bin);
fprintf('  block_dx    = %.4f m\n', scene.block_dx);
fprintf('  block_dy    = %.4f m\n', scene.block_dy);

%% ================= Step 3: Synthesize Full-Aperture HRRP =================
% 这里保持原来的物理回波链路：
% 多个散射点复数相干叠加生成 HRRP
HRRP_Full = synthesize_hrrp_from_physical_scatterers(radar, scatterers);

% 只做整体幅度归一化，不改变 RD 处理逻辑
HRRP_Full = HRRP_Full / (max(abs(HRRP_Full(:))) + eps);

%% ================= Step 4: Original Full-Aperture RD Image =================
% 正常 RD 处理：对 full-aperture HRRP 做方位向 FFT
% 不使用 Img_Ideal_Display，不使用二值散射点图
Img_Label = fftshift(fft_u(HRRP_Full, 2), 2);
Img_Label = Img_Label / (max(abs(Img_Label(:))) + eps);

%% ================= Step 5: Equal-Amplitude Scatterer Display =================
% 只用于论文展示真实散射点位置：
% 每个散射点等幅显示，不做复数相干叠加
[Img_Ideal_Display, collision_count] = rasterize_scatterers_equal_display(scatterers, radar);

Img_Ideal_Display = Img_Ideal_Display / (max(abs(Img_Ideal_Display(:))) + eps);

fprintf('\nIdeal scatterer display:\n');
fprintf('  collision_count = %d\n', collision_count);

%% ================= Step 6: Export Figures =================
pwd= 'D:\document\yanyi\电子书\论文\贝叶斯算法的寻优\manuscript\实验代码';
out_dir = fullfile(pwd, 'out_sim_figs');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

roi_full  = [1, radar.Nr, 1, radar.Na];    % [range_min, range_max, az_min, az_max]
dyn_range = -30;

% ---------- 1) 等幅散射点图：无 colorbar ----------
eps_scatter = fullfile(out_dir, 'sim_ideal_scatterer_equal_no_colorbar.eps');
png_scatter = fullfile(out_dir, 'sim_ideal_scatterer_equal_no_colorbar.png');

plot_isar_roi_eps_png_no_colorbar_xy( ...
    Img_Ideal_Display, ...
    eps_scatter, ...
    png_scatter, ...
    roi_full, ...
    dyn_range);

% ---------- 2) 原始 RD 图：正常 RD 处理，有 colorbar ----------
eps_rd = fullfile(out_dir, 'sim_full_aperture_RD_original.eps');
png_rd = fullfile(out_dir, 'sim_full_aperture_RD_original.png');

plot_isar_roi_eps_png_with_colorbar_xy( ...
    Img_Label, ...
    eps_rd, ...
    png_rd, ...
    roi_full, ...
    dyn_range);

% ---------- 保存数据 ----------
mat_out = fullfile(out_dir, 'sim_scatterer_RD_original_data.mat');

save(mat_out, ...
    'radar', ...
    'scene', ...
    'scatterers', ...
    'HRRP_Full', ...
    'Img_Ideal_Display', ...
    'Img_Label', ...
    'collision_count', ...
    'roi_full', ...
    'dyn_range');

fprintf('\nExport completed:\n');
fprintf('  Scatterer EPS : %s\n', eps_scatter);
fprintf('  Scatterer PNG : %s\n', png_scatter);
fprintf('  RD EPS        : %s\n', eps_rd);
fprintf('  RD PNG        : %s\n', png_rd);
fprintf('  MAT data      : %s\n', mat_out);
fprintf('\nDone.\n');

%% ================================================================
% Local Functions
% ================================================================

function scatterers = generate_aircraft_like_physical_scatterers(scene)
% 生成飞机样 block-sparse 物理散射中心
% 这里保持原来的散射点生成逻辑：
% - 每个 block 内 3 个散射点
% - 幅度固定为 0.8
% - 每个 block 有自己的 base_phase
% - 散射点位置带轻微扰动和轻微吸附
%
% 注意：
% 这里不要为了显示等幅而改变真实 scatterers。
% 显示等幅只在 rasterize_scatterers_equal_display 中完成。

rng(1);

local_patterns = {
    [0 0;  1 0;  0 1], ...
    [0 0; -1 0;  0 1], ...
    [0 0;  1 0;  0 -1], ...
    [0 0; -1 0;  0 -1], ...
    [0 0;  1 1; -1 0]
    };

% 12 个机身 block
y_body = linspace(-scene.body_len/2, scene.body_len/2, 12).';
x_body = 0.25 * scene.dx_bin * sin(linspace(-pi/2, pi/2, numel(y_body))).';

% 6 个左翼 + 6 个右翼
x_wing = linspace(2, scene.wing_span/2, 6).';
y_wing = scene.wing_y + 0.25 * x_wing;

x_left_wing  = -x_wing;
y_left_wing  = y_wing;

x_right_wing =  x_wing;
y_right_wing = y_wing;

% 3 个左尾翼 + 3 个右尾翼
x_tail = linspace(1.5, scene.tail_span/2, 3).';
y_tail = scene.tail_y + 0.35 * x_tail;

x_left_tail  = -x_tail;
y_left_tail  = y_tail;

x_right_tail =  x_tail;
y_right_tail = y_tail;

xc = [x_body; x_left_wing; x_right_wing; x_left_tail; x_right_tail];
yc = [y_body; y_left_wing; y_right_wing; y_left_tail; y_right_tail];

if numel(xc) ~= scene.numBlocks
    error('Block center count mismatch.');
end

scatterers = struct('x', {}, 'y', {}, 'sigma', {});
count = 0;

for b = 1:scene.numBlocks
    pat = local_patterns{randi(numel(local_patterns))};

    x0 = xc(b) + scene.jitter_center * scene.block_dx * randn();
    y0 = yc(b) + scene.jitter_center * scene.block_dy * randn();

    base_phase = 2 * pi * rand();

    for k = 1:scene.ptsPerBlock
        count = count + 1;

        xk = x0 + scene.block_dx * pat(k,1) + ...
            scene.jitter_point * scene.block_dx * randn();

        yk = y0 + scene.block_dy * pat(k,2) + ...
            scene.jitter_point * scene.block_dy * randn();

        % 按比例吸附到 RD 栅格
        if isfield(scene, 'snap_to_grid') && scene.snap_to_grid
            x_snap = round(xk / scene.dx_bin) * scene.dx_bin;
            y_snap = round(yk / scene.dy_bin) * scene.dy_bin;

            xk = (1 - scene.snap_ratio_x) * xk + scene.snap_ratio_x * x_snap;
            yk = (1 - scene.snap_ratio_y) * yk + scene.snap_ratio_y * y_snap;
        end

        % 保持原来的输入：幅度固定，block 内相位一致
        amp = 0.8 + 0.0 * rand();
        phs = base_phase + 0.00 * randn();
        sigma = amp * exp(1j * phs);

        scatterers(count).x = xk;
        scatterers(count).y = yk;
        scatterers(count).sigma = sigma;
    end
end
end


function HRRP = synthesize_hrrp_from_physical_scatterers(radar, scatterers)
% 按 HRRP 模型逐散射点合成 full-aperture 回波：
%
% h(t_hat, tm) = sum_q sigma'_q * sinc(B*(t_hat - 2*y_q/c))
%                            * exp(-j*4*pi*fc*x_q*omega_r*tm/c)
%
% 其中：
% sigma'_q = sigma_q * exp(-j*4*pi*fc*y_q/c)

Nr = radar.Nr;
Na = radar.Na;

t_fast = radar.t_fast(:);     % Nr x 1
tm     = radar.tm(:).';       % 1 x Na

HRRP = zeros(Nr, Na);

for q = 1:numel(scatterers)
    xq = scatterers(q).x;
    yq = scatterers(q).y;
    sigma_q = scatterers(q).sigma;

    % 距离传播相位
    sigma_prime = sigma_q .* exp(-1j * 4 * pi * radar.fc * yq / radar.c);

    % 距离压缩后的 sinc 响应
    tau_q = 2 * yq / radar.c;
    range_prof = sinc(radar.Bw * (t_fast - tau_q));

    % 方位慢时间相位
    az_phase = exp(-1j * 4 * pi * radar.fc * xq * radar.omega_r * tm / radar.c);

    % 复数相干叠加
    HRRP = HRRP + sigma_prime * (range_prof * az_phase);
end
end


function [Img_Display, collision_count] = rasterize_scatterers_equal_display(scatterers, radar)
% 论文展示用散射点图：
% 1) 只显示散射点位置
% 2) 每个散射点等幅显示
% 3) 不做复数相干叠加
% 4) 不影响 RD 图生成
%
% 输出：
% Img_Display(rr, dd) = 1 表示该 RD cell 有散射点

Img_Display = zeros(radar.Nr, radar.Na);
collision_count = 0;

for q = 1:numel(scatterers)
    yq = scatterers(q).y;
    xq = scatterers(q).x;

    % y -> range bin
    rCentered = round(yq / radar.delta_r);
    rr = rCentered + floor(radar.Nr / 2) + 1;

    % x -> Doppler frequency -> azimuth bin
    fD = -2 * xq * radar.omega_r / radar.lambda;
    dCentered = round(fD * radar.Na / radar.PRF);
    dd = dCentered + floor(radar.Na / 2) + 1;

    if rr >= 1 && rr <= radar.Nr && dd >= 1 && dd <= radar.Na
        if Img_Display(rr, dd) ~= 0
            collision_count = collision_count + 1;
        end

        % 等幅显示，只标位置
        Img_Display(rr, dd) = 1.0;
    end
end
end


function y = fft_u(x, dim)
% Unitary FFT
y = fft(x, [], dim) / sqrt(size(x, dim));
end


function y = ifft_u(x, dim)
% Unitary IFFT
y = ifft(x, [], dim) * sqrt(size(x, dim));
end


function plot_isar_roi_eps_png_no_colorbar_xy(img, eps_outname, png_outname, roi_1based, dyn_range)
% =========================================================================
% 散射点展示图输出函数：EPS + PNG，无 colorbar
%
% x 轴：Azimuth cell
% y 轴：Range cell
% 无 title
% 无 colorbar
% =========================================================================

if nargin < 5 || isempty(dyn_range)
    dyn_range = -30;
end

img_abs = abs(img);
[H, W] = size(img_abs);

if isempty(img_abs) || max(img_abs(:)) <= 0
    img_db = dyn_range * ones(H, W);
else
    img_db = 20 * log10(img_abs / (max(img_abs(:)) + eps) + eps);
    img_db = max(img_db, dyn_range);
end

% ROI
r_min = max(1, round(roi_1based(1)));
r_max = min(H, round(roi_1based(2)));
c_min = max(1, round(roi_1based(3)));
c_max = min(W, round(roi_1based(4)));

if r_min > r_max || c_min > c_max
    error('ROI 范围非法：range 或 azimuth 设置错误。');
end

img_roi_db = img_db(r_min:r_max, c_min:c_max);

x_axis = c_min:c_max;
y_axis = r_min:r_max;

figW = 520;
figH = 500;

fig = figure('Color', 'w', ...
    'Units', 'pixels', ...
    'Position', [100, 100, figW, figH], ...
    'Visible', 'off');

ax = axes('Parent', fig, ...
    'Units', 'normalized', ...
    'Position', [0.14, 0.14, 0.82, 0.82]);

imagesc(ax, x_axis, y_axis, img_roi_db);
axis(ax, 'normal');
set(ax, 'YDir', 'reverse');

colormap(ax, hot);
caxis(ax, [dyn_range, 0]);

xlabel(ax, 'Azimuth cell', ...
    'FontSize', 12, ...
    'FontWeight', 'normal');

ylabel(ax, 'Range cell', ...
    'FontSize', 12, ...
    'FontWeight', 'normal');

set(ax, ...
    'FontSize', 11, ...
    'LineWidth', 1.0, ...
    'TickDir', 'in', ...
    'TickLength', [0.015 0.015], ...
    'Box', 'on', ...
    'Layer', 'top');

xlim(ax, [c_min, c_max]);
ylim(ax, [r_min, r_max]);
set(ax, 'YDir', 'reverse');

% Tick 设置
set_axis_ticks_for_roi(ax, c_min, c_max, r_min, r_max);

set(fig, 'PaperPositionMode', 'auto');
set(fig, 'InvertHardcopy', 'off');
set(fig, 'Renderer', 'painters');

print(fig, eps_outname, '-depsc', '-painters', '-r600');
print(fig, png_outname, '-dpng', '-r600');

close(fig);
end


function plot_isar_roi_eps_png_with_colorbar_xy(img, eps_outname, png_outname, roi_1based, dyn_range)
% =========================================================================
% RD 图输出函数：EPS + PNG + colorbar
%
% x 轴：Azimuth cell
% y 轴：Range cell
% 无 title
% 有 colorbar
% =========================================================================

if nargin < 5 || isempty(dyn_range)
    dyn_range = -30;
end

img_abs = abs(img);
[H, W] = size(img_abs);

if isempty(img_abs) || max(img_abs(:)) <= 0
    img_db = dyn_range * ones(H, W);
else
    img_db = 20 * log10(img_abs / (max(img_abs(:)) + eps) + eps);
    img_db = max(img_db, dyn_range);
end

% ROI
r_min = max(1, round(roi_1based(1)));
r_max = min(H, round(roi_1based(2)));
c_min = max(1, round(roi_1based(3)));
c_max = min(W, round(roi_1based(4)));

if r_min > r_max || c_min > c_max
    error('ROI 范围非法：range 或 azimuth 设置错误。');
end

img_roi_db = img_db(r_min:r_max, c_min:c_max);

x_axis = c_min:c_max;
y_axis = r_min:r_max;

figW = 560;
figH = 500;

fig = figure('Color', 'w', ...
    'Units', 'pixels', ...
    'Position', [100, 100, figW, figH], ...
    'Visible', 'off');

ax = axes('Parent', fig, ...
    'Units', 'normalized', ...
    'Position', [0.12, 0.14, 0.72, 0.82]);

imagesc(ax, x_axis, y_axis, img_roi_db);
axis(ax, 'normal');
set(ax, 'YDir', 'reverse');

colormap(ax, hot);
caxis(ax, [dyn_range, 0]);

xlabel(ax, 'Azimuth cell', ...
    'FontSize', 12, ...
    'FontWeight', 'normal');

ylabel(ax, 'Range cell', ...
    'FontSize', 12, ...
    'FontWeight', 'normal');

set(ax, ...
    'FontSize', 11, ...
    'LineWidth', 1.0, ...
    'TickDir', 'in', ...
    'TickLength', [0.015 0.015], ...
    'Box', 'on', ...
    'Layer', 'top');

xlim(ax, [c_min, c_max]);
ylim(ax, [r_min, r_max]);
set(ax, 'YDir', 'reverse');

% Tick 设置
set_axis_ticks_for_roi(ax, c_min, c_max, r_min, r_max);

% Colorbar
cb = colorbar(ax, 'eastoutside');
cb.FontSize = 11;
cb.LineWidth = 1.0;
cb.Position = [0.87, 0.14, 0.03, 0.82];

if mod(abs(dyn_range), 5) == 0
    cb.Ticks = dyn_range:5:0;
else
    cb.Ticks = linspace(dyn_range, 0, 7);
end

set(fig, 'PaperPositionMode', 'auto');
set(fig, 'InvertHardcopy', 'off');
set(fig, 'Renderer', 'painters');

print(fig, eps_outname, '-depsc', '-painters', '-r600');
print(fig, png_outname, '-dpng', '-r600');

close(fig);
end


function set_axis_ticks_for_roi(ax, c_min, c_max, r_min, r_max)
% 根据 ROI 大小自动设置坐标刻度

x_span = c_max - c_min + 1;
y_span = r_max - r_min + 1;

if x_span > 120
    x_step = 50;
else
    x_step = 10;
end

if y_span > 120
    y_step = 50;
else
    y_step = 10;
end

xtick_start = ceil(c_min / x_step) * x_step;
xtick_end   = floor(c_max / x_step) * x_step;
ytick_start = ceil(r_min / y_step) * y_step;
ytick_end   = floor(r_max / y_step) * y_step;

if xtick_end >= xtick_start
    set(ax, 'XTick', xtick_start:x_step:xtick_end);
end

if ytick_end >= ytick_start
    set(ax, 'YTick', ytick_start:y_step:ytick_end);
end
end