function generate_dataset_sim_windowedRDGT_noiseCorrected()
% =========================================================================
% 面向 PyTorch / 深度展开训练的“纯仿真物理数据”生成器
%
% 设计目标：
% 1) 保留你第一个文件的数据集保存格式（train/val/test + sample_xxxxx.mat）
% 2) 不再读取实测 DAT/MAT，也不再从实测数据构造 pseudo GT
% 3) 每个样本都按第二个文件的“物理散射点 -> HRRP -> RD”链路独立生成
% 4) 关键修改：先对 clean HRRP/echo 加窗，再生成 full-aperture RD label
% 5) X_gt_norm 与 X_gt_raw_norm 均使用“加窗后的 full-aperture RD label”
% 6) Img_Ideal 仍然保存为 X_gt_ideal_norm，只用于对照，不作为当前训练监督
%
% 说明：
% - 如果你希望 X_gt_raw 也等于 Img_Ideal，只需把 sample 中那一行改成：
%       X_gt_raw_i = Img_Ideal;
% - 当前观测仍按“频域观测 + 稀疏采样 + 复高斯噪声”的格式保存，
%   尽量兼容你现有的 PyTorch 读取代码。
% =========================================================================

    clear; clc; close all;

    %% =========================
    % 0) 用户配置
    % =========================
    cfg = struct();

    % 输出路径
    cfg.save_root = 'D:\document\yanyi\电子书\论文\贝叶斯算法的寻优\2DFIRclaude\仿真数据训练\代码\sim_physical_windowedRDGT_dataset_noiseCorrected';

    % 数据量
    cfg.num_train = 800;
    cfg.num_val   = 100;
    cfg.num_test  = 100;

    % 随机种子
    cfg.seed = 2026;

    % ---------- SNR ----------
    cfg.snr_min = 0;
    cfg.snr_max = 10;
    cfg.noise_power_mode = 'mean';   % 'mean' | 'peak'

    % ---------- 掩膜：每样本独立 ----------
    cfg.use_random_mask = true;
    cfg.mask_keep_ratio_min = 0.25;
    cfg.mask_keep_ratio_max = 0.50;

    % ---------- 归一化 ----------
    cfg.scale_mode = 'obs_rms';      % 'gt_percentile' | 'gt_max' | 'obs_rms'
    cfg.scale_percentile = 99.5;
    cfg.scale_eps = 1e-8;

    % ---------- 是否画预览 ----------
    cfg.show_preview = true;

    % ---------- 保存加窗 full-aperture RD GT 预览图 ----------
    % 注意：这里预览的是 X_gt_raw_i / X_gt_used_i，也就是当前训练用的加窗 RD label。
    cfg.preview.enable = true;
    cfg.preview.visible = 'off';        % 'on' 弹窗；'off' 只保存
    cfg.preview.save_png = false;
    cfg.preview.save_eps = false;       % 大批量生成时建议 false，确认图没问题后再开
    cfg.preview.dyn_range = -30;
    cfg.preview.num_train = 3;          % train 前几个样本导出预览
    cfg.preview.num_val   = 2;          % val 前几个样本导出预览
    cfg.preview.num_test  = 2;          % test 前几个样本导出预览
    cfg.preview.output_dirname = 'preview_windowedRD';

    % ---------- 当前训练监督目标 ----------
    % windowed_full_rd: X_gt_norm = 加窗后的 full-aperture RD label
    cfg.target_mode = 'windowed_full_rd';

    % ---------- clean HRRP / echo 加窗，用于压制 RD 旁瓣 ----------
    cfg.window.enable = true;
    cfg.window.range_type = 'hamming';        % 'hamming' | 'hann' | 'blackman' | 'rect' | 'none'
    cfg.window.az_type    = 'hamming';        % 'hamming' | 'hann' | 'blackman' | 'rect' | 'none'
    cfg.window.normalize_peak = true;

    % ---------- 同步生成 6 个固定测试 MAT ----------
    % 6 个测试样本与训练样本共用同一个核心生成链路：
    % 物理散射点 -> HRRP -> 加窗 -> full RD GT -> 观测域加噪/稀疏
    cfg.generate_six_common = true;
    cfg.six_save_dir = fullfile(cfg.save_root, 'sim_six_common_mat_windowedRDGT');
    cfg.six_seed = 3407;
    cfg.six_snr_list  = [0, 5, 10];
    cfg.six_keep_list = [0.25, 0.50];
    cfg.six_same_scene = true;
    cfg.six_reuse_mask_by_keep = true;

    % ---------- 是否生成 teacher ----------
    cfg.generate_teacher = false;

    % ---------- 可选：加入轻微运动扰动 ----------
    cfg.enable_motion_perturb = false;
    cfg.motion_prob = 0.35;
    cfg.v_range = [-0.10, 0.10];
    cfg.a_range = [-0.01, 0.01];

    % ---------- 物理散射场参数（来自第二个文件思路） ----------
    cfg.scene.numBlocks   = 30;
    cfg.scene.ptsPerBlock = 3;
    cfg.scene.body_len    = 14;
    cfg.scene.wing_span   = 18;
    cfg.scene.tail_span   = 7;
    cfg.scene.wing_y      = 2.5;
    cfg.scene.tail_y      = -5.0;

    cfg.scene.snap_to_grid = false;   % 与原 RD 生图代码保持一致：不吸附到 RD 栅格
    cfg.scene.snap_ratio_x = 0.85;
    cfg.scene.snap_ratio_y = 0.85;
    cfg.scene.jitter_center = 0.08;
    cfg.scene.jitter_point  = 0.05;

    % 每个样本可随机浮动一点，增加多样性
    cfg.scene_randomize = true;
    cfg.scene.numBlocks_range = [24, 36];
    cfg.scene.body_len_range  = [12, 16];
    cfg.scene.wing_span_range = [16, 22];
    cfg.scene.tail_span_range = [5, 9];
    cfg.scene.wing_y_range    = [1.5, 3.5];
    cfg.scene.tail_y_range    = [-6.0, -4.0];
    cfg.scene.snap_ratio_range = [0.75, 0.95];
    cfg.scene.jitter_center_range = [0.04, 0.10];
    cfg.scene.jitter_point_range  = [0.03, 0.08];

    % 可选幅相扰动（直接作用于物理散射中心反射系数）
    cfg.sigma_amp_jitter_std   = 0.08;
    cfg.sigma_phase_jitter_std = 0.10;

    % 如果后面要生成 teacher，可在这里补 opts
    cfg.teacher_opts = struct();

    %% =========================
    % 1) 初始化
    % =========================
    rng(cfg.seed);

    if cfg.generate_teacher && ~exist('run_MC_2D_FIR_GEM', 'file')
        error('未找到 run_MC_2D_FIR_GEM.m，请先加入 MATLAB 路径。');
    end

    radar = init_radar_sim();
    total_num = cfg.num_train + cfg.num_val + cfg.num_test;

    prepare_save_dirs(cfg.save_root);

    fprintf('================ Physical Simulation Training Dataset Generator ================\n');
    fprintf('Step 1: 构造参考仿真样本（Img_Label / Img_Ideal / full-aperture HRRP）...\n');

    %% =========================
    % 2) 构造参考 base 样本
    % =========================
    [Y_clean_base, X_gt_raw_base, X_gt_used_base, clean_aux] = generate_physical_training_sample(radar, cfg);

    fprintf('  -> Y_clean_base   size = [%d, %d]\n', size(Y_clean_base,1), size(Y_clean_base,2));
    fprintf('  -> X_gt_raw_base  size = [%d, %d]\n', size(X_gt_raw_base,1), size(X_gt_raw_base,2));
    fprintf('  -> X_gt_used_base size = [%d, %d]\n', size(X_gt_used_base,1), size(X_gt_used_base,2));

    %% =========================
    % 3) 预览
    % =========================
    if cfg.show_preview
        show_preview_figure_sim(Y_clean_base, X_gt_raw_base, X_gt_used_base);
    end

    %% =========================
    % 4) 保存 meta
    % =========================
    dataset_meta = struct();
    dataset_meta.cfg = cfg;
    dataset_meta.radar = radar;
    dataset_meta.base_info = struct( ...
        'data_mode', 'pure_sim_physical', ...
        'note', ['纯仿真训练集：物理散射点 -> 原始尺度 HRRP -> 加窗 -> 原始尺度 full-aperture RD label；', ...
                 '噪声在 Y_clean=Phi_full(X_gt_raw) 的观测域按平均功率加入，随后再做方位向整列稀疏；', ...
                 '归一化只在加噪/mask 之后用于网络输入保存，Img_Ideal 仅作为 X_gt_ideal_norm 对照保存']);

    save(fullfile(cfg.save_root, 'dataset_meta.mat'), 'dataset_meta', '-v7');

    %% =========================
    % 5) 批量生成样本
    % =========================
    fprintf('Step 2: 开始生成 train / val / test 样本...\n');

    keep_ratio_hist = zeros(total_num, 1);
    snr_hist = zeros(total_num, 1);

    for i = 1:total_num

        % ---- 5.1 物理链路生成单样本 ----
        [Y_clean_i, X_gt_raw_i, X_gt_used_i, aug_info] = generate_physical_training_sample(radar, cfg);
        source_type = 'sim_physical_windowedRDGT_noiseCorrected';

        % ---- 5.2 可选：加入轻微运动扰动 ----
        motion_params = struct('enabled', false, 'v', 0.0, 'a', 0.0);
        if cfg.enable_motion_perturb && (rand() < cfg.motion_prob)
            motion_params.enabled = true;
            motion_params.v = rand_uniform(cfg.v_range(1), cfg.v_range(2));
            motion_params.a = rand_uniform(cfg.a_range(1), cfg.a_range(2));
            Y_clean_i = apply_motion_phase_error_fc(Y_clean_i, radar, motion_params.v, motion_params.a);
        end

        % ---- 5.3 加噪：在原始尺度 clean observation 上加复高斯噪声 ----
        snr_db = rand_uniform(cfg.snr_min, cfg.snr_max);
        Y_noisy = add_complex_noise(Y_clean_i, snr_db, cfg.noise_power_mode);

        if i == 1
            noise_i = Y_noisy - Y_clean_i;
            meas_snr_i = 10 * log10(mean(abs(Y_clean_i(:)).^2) / ...
                max(mean(abs(noise_i(:)).^2), eps));
            fprintf('Noise check sample 1: target SNR = %.2f dB, measured SNR = %.2f dB\n', ...
                snr_db, meas_snr_i);
            fprintf('Clean echo power: mean = %.6e, peak/mean = %.2f dB\n', ...
                mean(abs(Y_clean_i(:)).^2), ...
                10*log10(max(abs(Y_clean_i(:)).^2) / max(mean(abs(Y_clean_i(:)).^2), eps)));
        end

        % ---- 5.4 每样本独立 Mask ----
        [Mask_single, keep_ratio, pulse_mask] = generate_random_mask(radar, cfg);
        keep_ratio_hist(i) = keep_ratio;
        snr_hist(i) = snr_db;

        if i == 1
            fprintf('Mask size = [%d, %d], kept pulses = %d / %d\n', ...
                size(Mask_single,1), size(Mask_single,2), sum(pulse_mask), radar.Na);
        end

        % ---- 5.5 定义当前样本算子：与原第一份文件保持一致 ----
        PhiH = @(Y) fftshift(fft(ifft(Y, [], 1), [], 2), 2);
        Phi  = @(X) fft(ifft(ifftshift(X, 2), [], 2), [], 1) .* Mask_single;

        % ---- 5.6 观测 + 初值 ----
        Y_input = Y_noisy .* Mask_single;
        X0 = PhiH(Y_input);

        % ---- 5.7 归一化 ----
        scale = compute_scale(Y_input, X_gt_raw_i, cfg, Mask_single);

        Y_input_norm      = Y_input     / scale;
        Y_clean_norm      = Y_clean_i   / scale;
        X0_norm           = X0          / scale;
        X_gt_norm         = X_gt_used_i / scale;   % 当前监督目标：加窗 full-aperture RD label
        X_gt_raw_norm     = X_gt_raw_i  / scale;   % 与 X_gt_norm 相同，便于兼容旧读取逻辑
        X_gt_ideal_norm   = aug_info.Img_Ideal / scale; % 理想散射点图，仅作对照，不作为当前训练目标
        X_gt_abs_norm     = abs(X_gt_norm);
        X_gt_raw_abs_norm = abs(X_gt_raw_norm);
        window_info       = aug_info.window_info;

        % ---- 5.8 teacher（可选） ----
        if cfg.generate_teacher
            teacher_opts = cfg.teacher_opts;
            teacher_opts.Nobs = nnz(Mask_single);

            obs = Y_input_norm(Mask_single ~= 0);
            teacher_opts.gamma0 = 3.0 / max(mean(abs(obs).^2), 1e-6);
            teacher_opts.X0 = X0_norm;

            alpha_scalar = 1.0 / max(mean(abs(X0_norm(:)).^2), 1e-4);
            teacher_opts.alpha0 = alpha_scalar * ones(size(X0_norm));

            Phi_teacherH = @(Y) fftshift(fft(ifft(Y, [], 1), [], 2), 2);
            Phi_teacher  = @(X) fft(ifft(ifftshift(X, 2), [], 2), [], 1) .* Mask_single;

            [X_teacher_norm, hyper_teacher, info_teacher] = run_MC_2D_FIR_GEM( ...
                Y_input_norm, Phi_teacher, Phi_teacherH, teacher_opts);
        else
            X_teacher_norm = [];
            hyper_teacher = [];
            info_teacher = [];
        end

        % ---- 5.9 划分数据集 ----
        [split_name, idx_local] = get_split_name(i, cfg.num_train, cfg.num_val, cfg.num_test);
        save_dir = fullfile(cfg.save_root, split_name);
        filename = fullfile(save_dir, sprintf('sample_%05d.mat', idx_local));

        % ---- 5.10 保存 ----
        save(filename, ...
            'Y_input_norm', ...
            'Y_clean_norm', ...
            'X0_norm', ...
            'X_gt_norm', ...
            'X_gt_raw_norm', ...
            'X_gt_ideal_norm', ...
            'X_gt_abs_norm', ...
            'X_gt_raw_abs_norm', ...
            'X_teacher_norm', ...
            'Mask_single', ...
            'pulse_mask', ...
            'snr_db', ...
            'scale', ...
            'keep_ratio', ...
            'source_type', ...
            'motion_params', ...
            'aug_info', ...
            'window_info', ...
            'hyper_teacher', ...
            'info_teacher', ...
            '-v7');

        % ---- 5.11 可选：导出加窗 full-aperture RD GT 预览图 ----
        % 当前循环里没有 sample_idx 变量；用 idx_local 表示当前 split 内编号。
        maybe_export_windowed_rd_preview( ...
            X_gt_raw_i, cfg, split_name, idx_local, save_dir, filename);

        if mod(i, 50) == 0 || i == total_num
            fprintf('  已生成 %4d / %4d | src=%s | snr=%.2f dB | keep=%.3f\n', ...
                i, total_num, source_type, snr_db, keep_ratio);
        end
    end

    %% =========================
    % 6) 保存参考与统计
    % =========================
    generation_summary = struct();
    generation_summary.source_type       = 'sim_physical';
    generation_summary.snr_min_actual    = min(snr_hist);
    generation_summary.snr_max_actual    = max(snr_hist);
    generation_summary.keep_min_actual   = min(keep_ratio_hist);
    generation_summary.keep_max_actual   = max(keep_ratio_hist);
    generation_summary.keep_mean_actual  = mean(keep_ratio_hist);

    save(fullfile(cfg.save_root, 'groundtruth_reference.mat'), ...
        'Y_clean_base', 'X_gt_raw_base', 'X_gt_used_base', ...
        'clean_aux', 'radar', 'generation_summary', '-v7');

    %% =========================
    % 7) 额外生成 6 个固定测试 MAT
    % =========================
    if isfield(cfg, 'generate_six_common') && cfg.generate_six_common
        generate_six_common_windowedRDGT(radar, cfg);
    end

    fprintf('-------------------------------------------------------------------\n');
    fprintf('数据源类型    : sim_physical\n');
    fprintf('SNR 实际范围  : [%.2f, %.2f] dB\n', ...
        generation_summary.snr_min_actual, generation_summary.snr_max_actual);
    fprintf('采样率实际范围: [%.3f, %.3f], mean=%.3f\n', ...
        generation_summary.keep_min_actual, ...
        generation_summary.keep_max_actual, ...
        generation_summary.keep_mean_actual);
    fprintf('输出目录: %s\n', cfg.save_root);
    fprintf('===================================================================\n');
end


% =========================================================================
% 单样本：物理散射点 -> HRRP -> Img_Label / Img_Ideal
% 输出：
%   Y_clean_freq : 频域 clean observation（与原第一文件一致）
%   X_gt_raw     : Img_Label（full-aperture RD image）
%   X_gt_used    : Img_Label（当前训练监督目标，与 X_gt_raw 相同）
% =========================================================================
function [Y_clean_freq, X_gt_raw, X_gt_used, aux] = generate_physical_training_sample(radar, cfg, scatterers_in)

    if nargin >= 3 && ~isempty(scatterers_in)
        scatterers = scatterers_in;
        scene = [];
    else
        scene = sample_scene_cfg(cfg, radar);
        scatterers = generate_aircraft_like_physical_scatterers(scene);
        scatterers = jitter_scatterer_sigma(scatterers, cfg);
    end

    % ============================================================
    % 1) 物理散射点 -> 原始尺度 HRRP
    %    关键：这里不要做 max(abs(.)) 归一化。
    %    噪声功率要由后续原始尺度 Y_clean_freq 的平均功率决定。
    % ============================================================
    HRRP_full_raw = synthesize_hrrp_from_physical_scatterers(radar, scatterers);

    % ============================================================
    % 2) 对原始尺度 HRRP 加窗
    %    注意：窗函数 normalize_peak=true 只保证窗函数峰值为 1，
    %    不会把整幅 HRRP 或 RD label 重新缩放到最大值 1。
    % ============================================================
    [HRRP_win_raw, window_info] = apply_hrrp_window(HRRP_full_raw, cfg.window);

    % ============================================================
    % 3) 加窗后的 full-aperture RD 作为当前训练 GT
    %    关键：Img_Label_raw 保留原始幅度尺度，不在这里做 max 归一化。
    %    后面主循环会用同一个 scale 同时归一化 Y_input / X0 / X_gt。
    % ============================================================
    Img_Label_raw = fftshift(fft(HRRP_win_raw, [], 2), 2);

    % ============================================================
    % 4) 从成像域 GT 正演到观测域，保持与 PyTorch / MATLAB 算子一致
    %    X_gt_raw --Phi_full--> Y_clean_freq
    %    Y_clean_freq --PhiH--> X_gt_raw
    % ============================================================
    Phi_full = @(X) fft(ifft(ifftshift(X, 2), [], 2), [], 1);
    Y_clean_freq = Phi_full(Img_Label_raw);

    % 当前监督目标：原始尺度加窗 full-aperture RD label
    X_gt_raw  = Img_Label_raw;
    X_gt_used = Img_Label_raw;

    % ============================================================
    % 5) 理想散射点图：只用于显示/对照，不参与加噪和训练监督
    % ============================================================
    Img_Ideal = rasterize_physical_scatterers(scatterers, radar);
    Img_Ideal = Img_Ideal / (max(abs(Img_Ideal(:))) + eps);

    % ============================================================
    % 6) 辅助信息：raw 变量用于检查；归一化变量仅用于显示
    % ============================================================
    aux = struct();
    aux.scene = scene;
    aux.scatterers = scatterers;

    aux.HRRP_full_raw = HRRP_full_raw;
    aux.HRRP_win_raw  = HRRP_win_raw;
    aux.Img_Label_raw = Img_Label_raw;

    aux.HRRP_full = HRRP_full_raw / (max(abs(HRRP_full_raw(:))) + eps);
    aux.HRRP_win  = HRRP_win_raw  / (max(abs(HRRP_win_raw(:)))  + eps);
    aux.Img_Label = Img_Label_raw / (max(abs(Img_Label_raw(:))) + eps);
    aux.Img_Ideal = Img_Ideal;

    aux.window_info = window_info;
    aux.type = 'sim_physical_windowedRDGT_noiseCorrected_rawScale';
    aux.signal_power_clean_echo = mean(abs(Y_clean_freq(:)).^2);
    aux.peak_power_clean_echo = max(abs(Y_clean_freq(:)).^2);
    aux.peak_to_mean_echo_db = 10 * log10(aux.peak_power_clean_echo / max(aux.signal_power_clean_echo, eps));
end

% =========================================================================
% 雷达参数（按第二个文件的物理仿真风格）
% =========================================================================
function radar = init_radar_sim()
    radar = struct();
    radar.c   = 3e8;
    radar.fc  = 10e9;          % Hz
    radar.Bw  = 600e6;         % Hz
    radar.B   = radar.Bw;
    radar.PRF = 100;           % Hz
    radar.Nr  = 256;
    radar.Na  = 256;

    radar.lambda  = radar.c / radar.fc;
    radar.delta_r = radar.c / (2 * radar.Bw);
    radar.omega_r = 0.06;      % rad/s

    radar.t_fast = ((0:radar.Nr-1) - radar.Nr/2) / radar.Bw;
    radar.tm     = ((0:radar.Na-1) - radar.Na/2) / radar.PRF;

    % 兼容第一份文件中的相位补偿函数
    radar.delta_f  = radar.B / radar.Nr;
    radar.freq_vec = radar.fc + ((0:radar.Nr-1)' - radar.Nr/2) * radar.delta_f;
end


% =========================================================================
% 采样一个 scene 配置，增加样本多样性
% =========================================================================
function scene = sample_scene_cfg(cfg, radar)
    scene = cfg.scene;

    if cfg.scene_randomize
        scene.numBlocks   = randi(cfg.scene.numBlocks_range);
        scene.body_len    = rand_uniform(cfg.scene.body_len_range(1),  cfg.scene.body_len_range(2));
        scene.wing_span   = rand_uniform(cfg.scene.wing_span_range(1), cfg.scene.wing_span_range(2));
        scene.tail_span   = rand_uniform(cfg.scene.tail_span_range(1), cfg.scene.tail_span_range(2));
        scene.wing_y      = rand_uniform(cfg.scene.wing_y_range(1),    cfg.scene.wing_y_range(2));
        scene.tail_y      = rand_uniform(cfg.scene.tail_y_range(1),    cfg.scene.tail_y_range(2));
        scene.snap_ratio_x = rand_uniform(cfg.scene.snap_ratio_range(1), cfg.scene.snap_ratio_range(2));
        scene.snap_ratio_y = rand_uniform(cfg.scene.snap_ratio_range(1), cfg.scene.snap_ratio_range(2));
        scene.jitter_center = rand_uniform(cfg.scene.jitter_center_range(1), cfg.scene.jitter_center_range(2));
        scene.jitter_point  = rand_uniform(cfg.scene.jitter_point_range(1),  cfg.scene.jitter_point_range(2));
    end

    scene.dy_bin = radar.c / (2 * radar.Bw);
    scene.dx_bin = radar.lambda * radar.PRF / (2 * radar.omega_r * radar.Na);

    scene.block_dx = 0.8 * scene.dx_bin;
    scene.block_dy = 0.8 * scene.dy_bin;
end


% =========================================================================
% 创建目录
% =========================================================================
function prepare_save_dirs(save_root)
    if ~exist(save_root, 'dir')
        mkdir(save_root);
    end

    subdirs = {'train', 'val', 'test'};
    for k = 1:numel(subdirs)
        p = fullfile(save_root, subdirs{k});
        if ~exist(p, 'dir')
            mkdir(p);
        end
    end
end


% =========================================================================
% Aircraft-like physical scatterers（来自第二个文件的核心思路）
% =========================================================================
function scatterers = generate_aircraft_like_physical_scatterers(scene)

    local_patterns = {
        [0 0;  1 0;  0 1], ...
        [0 0; -1 0;  0 1], ...
        [0 0;  1 0;  0 -1], ...
        [0 0; -1 0;  0 -1], ...
        [0 0;  1 1; -1 0]  ...
    };

    % 将 numBlocks 按机身 / 左右翼 / 左右尾翼分配
    nb_body = max(8, round(scene.numBlocks * 0.40));
    nb_wing_each = max(4, round(scene.numBlocks * 0.20));
    nb_tail_each = max(2, floor((scene.numBlocks - nb_body - 2*nb_wing_each) / 2));

    used_blocks = nb_body + 2*nb_wing_each + 2*nb_tail_each;
    while used_blocks > scene.numBlocks && nb_body > 8
        nb_body = nb_body - 1;
        used_blocks = nb_body + 2*nb_wing_each + 2*nb_tail_each;
    end
    while used_blocks < scene.numBlocks
        nb_body = nb_body + 1;
        used_blocks = nb_body + 2*nb_wing_each + 2*nb_tail_each;
    end

    y_body = linspace(-scene.body_len/2, scene.body_len/2, nb_body).';
    x_body = 0.25 * scene.dx_bin * sin(linspace(-pi/2, pi/2, numel(y_body))).';

    x_wing = linspace(2, scene.wing_span/2, nb_wing_each).';
    y_wing = scene.wing_y + 0.25 * x_wing;

    x_left_wing  = -x_wing;
    y_left_wing  = y_wing;
    x_right_wing =  x_wing;
    y_right_wing =  y_wing;

    x_tail = linspace(1.5, scene.tail_span/2, nb_tail_each).';
    y_tail = scene.tail_y + 0.35 * x_tail;

    x_left_tail  = -x_tail;
    y_left_tail  = y_tail;
    x_right_tail =  x_tail;
    y_right_tail =  y_tail;

    xc = [x_body; x_left_wing; x_right_wing; x_left_tail; x_right_tail];
    yc = [y_body; y_left_wing; y_right_wing; y_left_tail; y_right_tail];

    scatterers = struct('x', {}, 'y', {}, 'sigma', {});
    count = 0;

    for b = 1:numel(xc)
        pat = local_patterns{randi(numel(local_patterns))};

        x0 = xc(b) + scene.jitter_center * scene.block_dx * randn();
        y0 = yc(b) + scene.jitter_center * scene.block_dy * randn();

        base_phase = 2 * pi * rand();

        for k = 1:scene.ptsPerBlock
            count = count + 1;

            xk = x0 + scene.block_dx * pat(k,1) + scene.jitter_point * scene.block_dx * randn();
            yk = y0 + scene.block_dy * pat(k,2) + scene.jitter_point * scene.block_dy * randn();

            if isfield(scene, 'snap_to_grid') && scene.snap_to_grid
                x_snap = round(xk / scene.dx_bin) * scene.dx_bin;
                y_snap = round(yk / scene.dy_bin) * scene.dy_bin;
                xk = (1 - scene.snap_ratio_x) * xk + scene.snap_ratio_x * x_snap;
                yk = (1 - scene.snap_ratio_y) * yk + scene.snap_ratio_y * y_snap;
            end

            amp = 0.8;
            phs = base_phase;
            sigma = amp * exp(1j * phs);

            scatterers(count).x = xk;
            scatterers(count).y = yk;
            scatterers(count).sigma = sigma;
        end
    end
end


% =========================================================================
% 对物理散射中心反射系数做轻微幅相扰动
% =========================================================================
function scatterers = jitter_scatterer_sigma(scatterers, cfg)
    for q = 1:numel(scatterers)
        amp_scale = 1 + cfg.sigma_amp_jitter_std * randn();
        amp_scale = max(amp_scale, 0.10);
        phs_scale = exp(1j * cfg.sigma_phase_jitter_std * randn());
        scatterers(q).sigma = scatterers(q).sigma * amp_scale * phs_scale;
    end
end


% =========================================================================
% 按论文 Section II 风格合成 HRRP
% =========================================================================
function HRRP = synthesize_hrrp_from_physical_scatterers(radar, scatterers)

    Nr = radar.Nr;
    Na = radar.Na;

    t_fast = radar.t_fast(:);
    tm = radar.tm(:).';

    HRRP = complex(zeros(Nr, Na));

    for q = 1:numel(scatterers)
        xq = scatterers(q).x;
        yq = scatterers(q).y;
        sigma_q = scatterers(q).sigma;

        sigma_prime = sigma_q .* exp(-1j * 4 * pi * radar.fc * yq / radar.c);

        tau_q = 2 * yq / radar.c;
        range_prof = sinc(radar.Bw * (t_fast - tau_q));
        az_phase = exp(-1j * 4 * pi * radar.fc * xq * radar.omega_r * tm / radar.c);

        HRRP = HRRP + sigma_prime * (range_prof * az_phase);
    end
end


% =========================================================================
% 将物理散射中心粗映射到 RD 栅格（就是第二个文件里的 Img_Ideal）
% =========================================================================
function Img = rasterize_physical_scatterers(scatterers, radar)
    Img = complex(zeros(radar.Nr, radar.Na));

    for q = 1:numel(scatterers)
        yq = scatterers(q).y;
        xq = scatterers(q).x;
        sigma_q = scatterers(q).sigma;

        rCentered = round(yq / radar.delta_r);
        rr = rCentered + floor(radar.Nr / 2) + 1;

        fD = -2 * xq * radar.omega_r / radar.lambda;
        dCentered = round(fD * radar.Na / radar.PRF);
        dd = dCentered + floor(radar.Na / 2) + 1;

        if rr >= 1 && rr <= radar.Nr && dd >= 1 && dd <= radar.Na
            Img(rr, dd) = Img(rr, dd) + sigma_q;
        end
    end
end


% =========================================================================
% 每样本独立随机 Mask（方位向整列采样）
% =========================================================================
function [Mask_single, keep_ratio, pulse_mask] = generate_random_mask(radar, cfg)
    if cfg.use_random_mask
        keep_ratio_target = rand_uniform(cfg.mask_keep_ratio_min, cfg.mask_keep_ratio_max);
        num_keep = round(radar.Na * keep_ratio_target);
        num_keep = max(1, min(radar.Na, num_keep));
        keep_cols = randperm(radar.Na, num_keep);

        pulse_mask = zeros(1, radar.Na);
        pulse_mask(keep_cols) = 1;
        Mask_single = repmat(pulse_mask, radar.Nr, 1);
        keep_ratio = num_keep / radar.Na;
    else
        pulse_mask = ones(1, radar.Na);
        Mask_single = ones(radar.Nr, radar.Na);
        keep_ratio = 1.0;
    end
end


% =========================================================================
% 复高斯加噪
% =========================================================================
function Y_noisy = add_complex_noise(Y_clean_like, snr_db, power_mode)
    switch lower(power_mode)
        case 'mean'
            sig_power = mean(abs(Y_clean_like(:)).^2);
        case 'peak'
            sig_power = max(abs(Y_clean_like(:)).^2);
        otherwise
            error('未知噪声功率模式: %s', power_mode);
    end

    noise_power = sig_power / (10^(snr_db / 10));
    noise = sqrt(noise_power / 2) * ...
        (randn(size(Y_clean_like)) + 1j * randn(size(Y_clean_like)));

    Y_noisy = Y_clean_like + noise;
end


% =========================================================================
% 归一化尺度
% =========================================================================
function scale = compute_scale(Y_input, X_gt_raw, cfg, Mask_single)
    switch lower(cfg.scale_mode)
        case 'gt_percentile'
            scale = prctile(abs(X_gt_raw(:)), cfg.scale_percentile);
        case 'gt_max'
            scale = max(abs(X_gt_raw(:)));
        case 'obs_rms'
            scale = norm(Y_input(:), 'fro') / sqrt(sum(Mask_single(:)));
        otherwise
            error('未知归一化模式: %s', cfg.scale_mode);
    end

    scale = max(scale, cfg.scale_eps);
end


% =========================================================================
% 轻微运动扰动：频域数据上的慢时间二次相位
% =========================================================================
function Y_out = apply_motion_phase_error_fc(Y_in, radar, v, a)
    t = radar.tm(:).';
    r_t = v * t + 0.5 * a * (t.^2);
    phase = exp(-1i * 4 * pi * radar.fc / radar.c * r_t);
    Y_out = Y_in .* repmat(phase, size(Y_in, 1), 1);
end


% =========================================================================
% 划分 train / val / test
% =========================================================================
function [split_name, idx_local] = get_split_name(i, ntrain, nval, ntest)
    if i <= ntrain
        split_name = 'train';
        idx_local = i;
    elseif i <= ntrain + nval
        split_name = 'val';
        idx_local = i - ntrain;
    elseif i <= ntrain + nval + ntest
        split_name = 'test';
        idx_local = i - ntrain - nval;
    else
        error('索引越界');
    end
end


% =========================================================================
% 预览图
% =========================================================================
function show_preview_figure_sim(Y_clean_freq, X_gt_raw, X_gt_used)
    % 与 Y_clean_freq = fft(HRRP_win_raw, [], 1) 对应，预览 HRRP 需做 ifft
    HRRP_full = ifft(Y_clean_freq, [], 1);

    figure('Name', 'Physical-Sim Training Dataset Preview', 'Color', 'w', ...
           'Position', [100, 100, 1500, 420]);

    ref_gt = max([abs(X_gt_raw(:)); abs(X_gt_used(:))]) + eps;

    subplot(1,4,1);
    imagesc(abs(HRRP_full));
    axis image; axis xy;
    colormap(gca, hot); colorbar;
    title('Full-Aperture HRRP |H|');
    xlabel('Azimuth pulse'); ylabel('Range bin');

    subplot(1,4,2);
    show_image_paper_style(abs(X_gt_raw), ref_gt, -35, 'hot');
    title('X\_gt\_raw = Windowed Full RD');

    subplot(1,4,3);
    show_image_paper_style(abs(X_gt_used), ref_gt, -35, 'hot');
    title('X\_gt\_used = Windowed Full RD');

    subplot(1,4,4);
    show_image_paper_style(abs(X_gt_raw - X_gt_used), [], -35, 'hot');
    title('|X\_gt\_raw - X\_gt\_used|');
end


% =========================================================================
% 论文风格显示
% =========================================================================
function show_image_paper_style(Img, ref_max, db_floor, cmap_name)
    if nargin < 2 || isempty(ref_max)
        ref_max = max(abs(Img(:))) + eps;
    end
    if nargin < 3 || isempty(db_floor)
        db_floor = -35;
    end
    if nargin < 4 || isempty(cmap_name)
        cmap_name = 'hot';
    end

    mag = abs(Img);
    img_db = 20 * log10(mag / ref_max + eps);
    img_db = max(img_db, db_floor);

    imagesc(img_db, [db_floor, 0]);
    axis image; axis xy;
    colorbar;
    xlabel('Azimuth bin');
    ylabel('Range bin');
    set(gca, 'Color', 'k');

    switch lower(cmap_name)
        case 'gray'
            colormap(gca, gray);
        case 'parula'
            colormap(gca, parula);
        otherwise
            colormap(gca, hot);
    end
end



% =========================================================================
% 对 full-aperture HRRP / echo 加窗
% =========================================================================
function [HRRP_win, window_info] = apply_hrrp_window(HRRP_full, win_cfg)
    [Nr, Na] = size(HRRP_full);

    if ~isfield(win_cfg, 'enable') || ~win_cfg.enable
        wr = ones(Nr, 1);
        wa = ones(1, Na);
        HRRP_win = HRRP_full;
    else
        wr = make_1d_window(Nr, win_cfg.range_type);
        wa = make_1d_window(Na, win_cfg.az_type).';

        if isfield(win_cfg, 'normalize_peak') && win_cfg.normalize_peak
            wr = wr / (max(abs(wr(:))) + eps);
            wa = wa / (max(abs(wa(:))) + eps);
        end

        W2 = wr * wa;
        HRRP_win = HRRP_full .* W2;
    end

    window_info = struct();
    window_info.enable = isfield(win_cfg, 'enable') && win_cfg.enable;
    window_info.range_type = win_cfg.range_type;
    window_info.az_type = win_cfg.az_type;
    window_info.range_window = wr;
    window_info.az_window = wa;
end


% =========================================================================
% 生成一维窗函数，不依赖额外工具箱
% =========================================================================
function w = make_1d_window(N, type_name)
    n = (0:N-1).';

    switch lower(type_name)
        case {'rect', 'rectangular', 'none'}
            w = ones(N, 1);
        case {'hann', 'hanning'}
            if N == 1
                w = 1;
            else
                w = 0.5 - 0.5 * cos(2*pi*n/(N-1));
            end
        case 'hamming'
            if N == 1
                w = 1;
            else
                w = 0.54 - 0.46 * cos(2*pi*n/(N-1));
            end
        case 'blackman'
            if N == 1
                w = 1;
            else
                w = 0.42 - 0.5*cos(2*pi*n/(N-1)) + 0.08*cos(4*pi*n/(N-1));
            end
        otherwise
            error('未知窗函数类型: %s', type_name);
    end
end


% =========================================================================
% 生成固定 keep_ratio 的方位向整列 Mask
% =========================================================================
function [Mask_single, actual_keep_ratio, pulse_mask] = generate_fixed_keep_mask(radar, keep_ratio_target)
    num_keep = round(radar.Na * keep_ratio_target);
    num_keep = max(1, min(radar.Na, num_keep));

    keep_cols = randperm(radar.Na, num_keep);
    pulse_mask = zeros(1, radar.Na);
    pulse_mask(keep_cols) = 1;

    Mask_single = repmat(pulse_mask, radar.Nr, 1);
    actual_keep_ratio = num_keep / radar.Na;
end


% =========================================================================
% 生成 6 个固定仿真测试 MAT：SNR {0,5,10} × keep {0.25,0.50}
% 与训练数据共用完全相同的核心生成链路。
% =========================================================================
function generate_six_common_windowedRDGT(radar, cfg)

    fprintf('\n================ Generate six common simulation MAT files ================\n');

    if ~exist(cfg.six_save_dir, 'dir')
        mkdir(cfg.six_save_dir);
    end

    rng(cfg.six_seed);

    % 固定同一个目标场景，便于 6 个条件横向对比
    if cfg.six_same_scene
        scene = sample_scene_cfg(cfg, radar);
        scatterers_fixed = generate_aircraft_like_physical_scatterers(scene);
        scatterers_fixed = jitter_scatterer_sigma(scatterers_fixed, cfg);
        [Y_clean_fixed, X_gt_raw_fixed, X_gt_used_fixed, aug_info_fixed] = ...
            generate_physical_training_sample(radar, cfg, scatterers_fixed);
    else
        scatterers_fixed = [];
        [Y_clean_fixed, X_gt_raw_fixed, X_gt_used_fixed, aug_info_fixed] = ...
            generate_physical_training_sample(radar, cfg, scatterers_fixed);
    end

    source_type = 'sim_physical_windowedRDGT_noiseCorrected_six_common';

    fprintf('Radar parameters:\n');
    fprintf('  fc       = %.2f GHz\n', radar.fc / 1e9);
    fprintf('  Bw       = %.1f MHz\n', radar.Bw / 1e6);
    fprintf('  PRF      = %.1f Hz\n', radar.PRF);
    fprintf('  Nr x Na  = %d x %d\n', radar.Nr, radar.Na);
    fprintf('  delta_r  = %.4f m\n', radar.delta_r);
    fprintf('GT mode:\n');
    fprintf('  X_gt_norm = windowed full-aperture RD label\n');
    fprintf('Window:\n');
    fprintf('  range = %s, azimuth = %s\n', cfg.window.range_type, cfg.window.az_type);

    % 如果设置复用 mask，则同一个 keep_ratio 在不同 SNR 下使用同一个 mask
    mask_bank = containers.Map('KeyType', 'char', 'ValueType', 'any');
    pulse_bank = containers.Map('KeyType', 'char', 'ValueType', 'any');
    keep_bank = containers.Map('KeyType', 'char', 'ValueType', 'any');

    sample_idx = 0;

    for is = 1:numel(cfg.six_snr_list)
        snr_db = cfg.six_snr_list(is);

        for ik = 1:numel(cfg.six_keep_list)
            keep_target = cfg.six_keep_list(ik);
            sample_idx = sample_idx + 1;

            % 每个条件噪声可复现；在原始尺度 clean observation 上加噪
            rng(cfg.six_seed + 1000 + sample_idx);
            Y_noisy = add_complex_noise(Y_clean_fixed, snr_db, cfg.noise_power_mode);

            key = sprintf('keep_%03d', round(keep_target * 1000));
            if cfg.six_reuse_mask_by_keep && isKey(mask_bank, key)
                Mask_single = mask_bank(key);
                pulse_mask = pulse_bank(key);
                keep_ratio = keep_bank(key);
            else
                rng(cfg.six_seed + 2000 + ik);
                [Mask_single, keep_ratio, pulse_mask] = generate_fixed_keep_mask(radar, keep_target);
                if cfg.six_reuse_mask_by_keep
                    mask_bank(key) = Mask_single;
                    pulse_bank(key) = pulse_mask;
                    keep_bank(key) = keep_ratio;
                end
            end

            % 算子与训练样本完全一致
            PhiH = @(Y) fftshift(fft(ifft(Y, [], 1), [], 2), 2);
            Phi  = @(X) fft(ifft(ifftshift(X, 2), [], 2), [], 1) .* Mask_single; %#ok<NASGU>

            Y_input = Y_noisy .* Mask_single;
            X0 = PhiH(Y_input);

            scale = compute_scale(Y_input, X_gt_raw_fixed, cfg, Mask_single);

            Y_input_norm      = Y_input / scale;
            Y_clean_norm      = Y_clean_fixed / scale;
            X0_norm           = X0 / scale;
            X_gt_norm         = X_gt_used_fixed / scale;
            X_gt_raw_norm     = X_gt_raw_fixed / scale;
            X_gt_ideal_norm   = aug_info_fixed.Img_Ideal / scale;
            X_gt_abs_norm     = abs(X_gt_norm);
            X_gt_raw_abs_norm = abs(X_gt_raw_norm);
            X_teacher_norm    = [];
            hyper_teacher     = [];
            info_teacher      = [];
            motion_params     = struct('enabled', false, 'v', 0.0, 'a', 0.0);
            aug_info          = aug_info_fixed;
            window_info       = aug_info.window_info;

            filename = fullfile(cfg.six_save_dir, sprintf( ...
                'sample_%05d_snr%02d_keep%02d.mat', ...
                sample_idx, round(snr_db), round(keep_ratio * 100)));

            save(filename, ...
                'Y_input_norm', ...
                'Y_clean_norm', ...
                'X0_norm', ...
                'X_gt_norm', ...
                'X_gt_raw_norm', ...
                'X_gt_ideal_norm', ...
                'X_gt_abs_norm', ...
                'X_gt_raw_abs_norm', ...
                'X_teacher_norm', ...
                'Mask_single', ...
                'pulse_mask', ...
                'snr_db', ...
                'scale', ...
                'keep_ratio', ...
                'source_type', ...
                'motion_params', ...
                'aug_info', ...
                'window_info', ...
                'hyper_teacher', ...
                'info_teacher', ...
                '-v7');

            % 六个固定测试样本全部导出 GT 预览，便于确认是否共用同一个目标
            maybe_export_windowed_rd_preview_force( ...
                X_gt_raw_fixed, cfg, cfg.six_save_dir, filename);

            fprintf('  Saved %s | snr=%.2f dB | keep=%.4f\n', ...
                filename, snr_db, keep_ratio);
        end
    end

    six_meta = struct();
    six_meta.radar = radar;
    six_meta.cfg = cfg;
    six_meta.note = 'Six common MAT files generated with the same noise-corrected raw-scale windowed-RD-GT chain as training data.';
    six_meta.gt_mode = 'X_gt_norm = windowed full-aperture RD label';
    save(fullfile(cfg.six_save_dir, 'six_common_meta.mat'), 'six_meta', '-v7');

    fprintf('Six common simulation MAT saved to:\n  %s\n', cfg.six_save_dir);
    fprintf('=======================================================================\n\n');
end

% =========================================================================
% 可选：按 train / val / test 前若干样本导出加窗 RD GT 预览
% =========================================================================
function maybe_export_windowed_rd_preview(img_rd, cfg, split_name, idx_local, save_dir, filename)

    if ~isfield(cfg, 'preview') || ~isfield(cfg.preview, 'enable') || ~cfg.preview.enable
        return;
    end

    do_preview = false;
    switch lower(split_name)
        case 'train'
            do_preview = (idx_local <= cfg.preview.num_train);
        case 'val'
            do_preview = (idx_local <= cfg.preview.num_val);
        case 'test'
            do_preview = (idx_local <= cfg.preview.num_test);
        otherwise
            do_preview = false;
    end

    if ~do_preview
        return;
    end

    maybe_export_windowed_rd_preview_force(img_rd, cfg, save_dir, filename);
end


% =========================================================================
% 强制导出一张加窗 RD GT 预览图
% =========================================================================
function maybe_export_windowed_rd_preview_force(img_rd, cfg, save_dir, filename)

    if ~isfield(cfg, 'preview') || ~isfield(cfg.preview, 'enable') || ~cfg.preview.enable
        return;
    end

    preview_dir = fullfile(save_dir, cfg.preview.output_dirname);
    if ~exist(preview_dir, 'dir')
        mkdir(preview_dir);
    end

    [~, sample_name_noext, ~] = fileparts(filename);
    png_out = fullfile(preview_dir, [sample_name_noext, '_windowedRDGT.png']);
    eps_out = fullfile(preview_dir, [sample_name_noext, '_windowedRDGT.eps']);

    plot_windowed_rd_preview( ...
        img_rd, ...
        png_out, ...
        eps_out, ...
        cfg.preview.dyn_range, ...
        cfg.preview.visible, ...
        cfg.preview.save_png, ...
        cfg.preview.save_eps);
end


% =========================================================================
% 显示 / 导出加窗后的 full-aperture RD label
% =========================================================================
function plot_windowed_rd_preview(img, png_out, eps_out, dyn_range, visible_mode, save_png, save_eps)

    if nargin < 4 || isempty(dyn_range)
        dyn_range = -30;
    end
    if nargin < 5 || isempty(visible_mode)
        visible_mode = 'off';
    end
    if nargin < 6 || isempty(save_png)
        save_png = true;
    end
    if nargin < 7 || isempty(save_eps)
        save_eps = false;
    end

    img_abs = abs(img);

    if isempty(img_abs) || max(img_abs(:)) <= 0
        img_db = dyn_range * ones(size(img_abs));
    else
        img_db = 20 * log10(img_abs / (max(img_abs(:)) + eps) + eps);
        img_db = max(img_db, dyn_range);
    end

    [H, W] = size(img_db);
    x_axis = 1:W;
    y_axis = 1:H;

    fig = figure('Color', 'w', ...
        'Units', 'pixels', ...
        'Position', [100, 100, 560, 500], ...
        'Visible', visible_mode);

    ax = axes('Parent', fig, ...
        'Units', 'normalized', ...
        'Position', [0.12, 0.14, 0.72, 0.82]);

    imagesc(ax, x_axis, y_axis, img_db);
    axis(ax, 'image');
    set(ax, 'YDir', 'reverse');

    colormap(ax, hot);
    caxis(ax, [dyn_range 0]);

    xlabel(ax, 'Azimuth cell', 'FontSize', 12, 'FontWeight', 'normal');
    ylabel(ax, 'Range cell',   'FontSize', 12, 'FontWeight', 'normal');
    title(ax, 'Windowed Full-Aperture RD Label', ...
        'FontSize', 12, 'FontWeight', 'normal');

    set(ax, ...
        'FontSize', 11, ...
        'LineWidth', 1.0, ...
        'TickDir', 'in', ...
        'TickLength', [0.015 0.015], ...
        'Box', 'on', ...
        'Layer', 'top');

    set_auto_ticks_preview(ax, W, H);

    cb = colorbar(ax, 'eastoutside');
    cb.FontSize = 11;
    cb.LineWidth = 1.0;
    if dyn_range == -30
        cb.Ticks = [-30 -25 -20 -15 -10 -5 0];
    else
        cb.Ticks = dyn_range:5:0;
    end
    cb.Position = [0.87, 0.14, 0.03, 0.82];

    set(fig, 'PaperPositionMode', 'auto');
    set(fig, 'InvertHardcopy', 'off');
    set(fig, 'Renderer', 'painters');

    if save_png
        print(fig, png_out, '-dpng', '-r300');
    end

    if save_eps
        print(fig, eps_out, '-depsc', '-painters', '-r600');
    end

    if strcmpi(visible_mode, 'off')
        close(fig);
    end
end


function set_auto_ticks_preview(ax, W, H)

    if W > 180
        x_step = 50;
    elseif W > 90
        x_step = 20;
    else
        x_step = 10;
    end

    if H > 180
        y_step = 50;
    elseif H > 90
        y_step = 20;
    else
        y_step = 10;
    end

    set(ax, 'XTick', 1:x_step:W);
    set(ax, 'YTick', 1:y_step:H);
end


% =========================================================================
% 工具函数
% =========================================================================
function x = rand_uniform(a, b)
    x = a + (b - a) * rand();
end
